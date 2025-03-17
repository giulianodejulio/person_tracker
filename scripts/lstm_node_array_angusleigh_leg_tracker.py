#!/usr/bin/env python3

import rospy
import rospkg  # to load ROS package paths dynamically
import numpy as np
from collections import deque  # to have fixed-size moving window
import argparse  # to define some default args for self.predictor() to work

from visualization_msgs.msg import Marker  # to handle /visualization_markers messages
from geometry_msgs.msg import Point  # to display predicted trajectory in RViz
from hrii_person_tracker.msg import PathArray  # format of output trajectories

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path  # for predicted trajectories

import tf

import threading
import time

# LSTM imports
import trajnetplusplustools
import trajnetbaselines.lstm.trajnet_evaluator as te  # to use LSTM model from Trajnet++
import trajnetbaselines.lstm.utils
import torch  # to use tensors

# dynamic ROS parameters
from dynamic_reconfigure.server import Server
from hrii_person_tracker.cfg import LstmNodeConfig

# CollisionDetection imports
from leg_tracker.msg import People


def cfg_callback(config, level):
  return config

class LstmNodeArray:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args, _ = add_arguments(self.parser)

        reconfigure_server = Server(LstmNodeConfig, callback=cfg_callback)

        self.thread_period = rospy.get_param("~lstm_node_loop_period")
        self.data_acquisition_period = rospy.Duration(self.thread_period)  # default value: 1 second
        self.deprecated_data_time = rospy.Duration(2*self.thread_period)

        # self.intimate_distance_threshold = 1.0
        self.obs_length = 9

        # Store a fixed-size moving window for each pedestrian in a dictionary of dictionaries.
        # pedestrian_data =
        # {0: {'x': (x0_{k-self.obs_length-1},..., x0_k), 'y': (y0_{k-self.obs_length-1},..., y0_k), 'last_update_time': ...},
        #  1: {'x': (x1_{k-self.obs_length-1},..., x1_k), 'y': (y1_{k-self.obs_length-1},..., y1_k), 'last_update_time': ...},
        #  ...
        # }
        self.pedestrian_data = {}
        self.data_lock = threading.Lock()
                
        # Dynamically get the model path
        package_path = rospkg.RosPack().get_path('hrii_person_tracker')
        self.model = f"{package_path}/scripts/{self.args.neural_network_model}_{self.args.interaction_module}_None.pkl"

        # Load LSTM model using Trajnet++
        self.predictor = te.load_predictor(self.model)

        # tf listener to get transform of odom frame (odom) wrt global frame (map) (because leg_detector works with odom frame)
        self.mobile_base  = rospy.get_param("~mobile_base")
        self.global_frame = self.mobile_base + "_map"
        self.odom_frame   = self.mobile_base + "_odom"
        self.listener     = tf.TransformListener()

        # Subscriber for visualization markers
        self.visualization_marker_sub = rospy.Subscriber('/visualization_marker', Marker, self.visualization_marker_callback)

        # Dynamic publishers for each pedestrian based on their ID. They will publish observed and predicted trajectories, respectfully.
        self.obs_traj_pubs = {}
        self.pred_traj_pubs = {}

        # Create a single publisher for PathArray
        self.predicted_trajectory_pub = rospy.Publisher('/predicted_trajectory', PathArray, queue_size=10)

        # Collision Detection ----------------------------------------------------------------------------------------
        # self.people_velocities = {}
        # self.path_array_msg = PathArray() # I save the prediction msg to use it in CollisionDetection
        # self.people_sub_ = rospy.Subscriber('/people', People, self.people_callback)
        # ------------------------------------------------------------------------------------------------------------

        # Start prediction thread
        self.prediction_thread = threading.Thread(target=self.process_data)
        self.prediction_thread.start()

    def create_publisher(self, pedestrian_id):
        """Create a publisher for a pedestrian's predicted trajectory if it does not exist."""
        # if pedestrian_id not in self.pred_traj_pubs:
        #     topic_name = f'/predicted_trajectory{pedestrian_id}'
        #     topic_name_obs = f'/observed_trajectory{pedestrian_id}'
        #     self.pred_traj_pubs[pedestrian_id] = rospy.Publisher(topic_name, Path, queue_size=10)
        #     self.obs_traj_pubs[pedestrian_id] = rospy.Publisher(topic_name_obs, Path, queue_size=10)
        #     rospy.loginfo(f"Created pub {pedestrian_id}")
            
        if pedestrian_id not in self.pred_traj_pubs:
            topic_name = f'/predicted_trajectory{pedestrian_id}'
            self.pred_traj_pubs[pedestrian_id] = rospy.Publisher(topic_name, Path, queue_size=10)

        if pedestrian_id not in self.obs_traj_pubs:
            topic_name_obs = f'/observed_trajectory{pedestrian_id}'
            self.obs_traj_pubs[pedestrian_id] = rospy.Publisher(topic_name_obs, Path, queue_size=10)

    def prediction_to_path(self, predicted_trajectory):
        """Convert predicted trajectory to nav_msgs/Path."""
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.global_frame

        for point in predicted_trajectory:
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = self.global_frame
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0  # 2D prediction => z = 0
            pose.pose.orientation.w = 1.0  # no orientation considered
            path.poses.append(pose)

        return path

    # def visualization_marker_callback_original(self, msg):
    #     if msg.ns == "People_tracked" and msg.points:
    #         with self.data_lock:
    #             # Get the position of the center of the legs in odom frame
    #             x_odom = 0.5*(msg.points[0].x + msg.points[1].x)
    #             y_odom = 0.5*(msg.points[0].y + msg.points[1].y)

    #             # Get the transform from odom frame to global frame
    #             try:
    #                 now = rospy.Time.now()
    #                 self.listener.waitForTransform(self.global_frame, self.odom_frame, now, rospy.Duration(1.0))
    #                 (trans, rot) = self.listener.lookupTransform(self.global_frame, self.odom_frame, now)
    #             except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #                 rospy.logerr("TF Exception occurred")
    #                 return

    #             # Create a transformation matrix from the obtained translation and rotation
    #             transform_matrix = tf.transformations.quaternion_matrix(rot)
    #             transform_matrix[0:3, 3] = trans

    #             # Convert the odom coordinates to homogeneous coordinates
    #             point_local = np.array([x_odom, y_odom, 0, 1])

    #             # Transform the point from the odom frame to the global frame
    #             point_global = np.dot(transform_matrix, point_local)
    #             x_global, y_global = point_global[0], point_global[1]

    #             # pedestrian_data is void: start filling the first position with data coming from the msg
    #             if not self.pedestrian_data:
    #                 rospy.loginfo("One ped detected. Id: 0")
    #                 self.pedestrian_data[0] = {
    #                         'x': deque(maxlen=self.obs_length),
    #                         'y': deque(maxlen=self.obs_length),
    #                         # 'window_full': False,
    #                         'last_update_time': rospy.Time.now()
    #                     }
    #                 while len(self.pedestrian_data[0]['x']) < self.obs_length:
    #                     self.pedestrian_data[0]['x'].append(x_global)
    #                     self.pedestrian_data[0]['y'].append(y_global)
    #                     # self.pedestrian_data[0]['window_full'] = True
    #                     self.pedestrian_data[0]['last_update_time']: rospy.Time.now()

    #             else:
    #                 closest_ped_id, min_distance = self.find_closest_pedestrian(x_global, y_global)

    #                 # A new pedestrian is detected: collect data as fast as possible, until the window is full, to start the prediction.
    #                 if min_distance > self.intimate_distance_threshold:
    #                     first_free_id = self.first_free_id()
    #                     rospy.loginfo("New ped detected. Id: %d", first_free_id)
    #                     self.pedestrian_data[first_free_id] = {
    #                         'x': deque(maxlen=self.obs_length),
    #                         'y': deque(maxlen=self.obs_length),
    #                         'last_update_time': rospy.Time.now()
    #                     }
    #                     while len(self.pedestrian_data[first_free_id]['x']) < self.obs_length:
    #                         self.pedestrian_data[first_free_id]['x'].append(x_global)
    #                         self.pedestrian_data[first_free_id]['y'].append(y_global)
    #                     self.pedestrian_data[first_free_id]['last_update_time'] = rospy.Time.now()
                        
    #                     rospy.loginfo("Current pedestrian IDs: %s", list(self.pedestrian_data.keys()))


    #                 # Append the new point to trajectory of closest_ped_id every self.data_acquisition_period seconds
    #                 elif min_distance <= self.intimate_distance_threshold and rospy.Time.now() - self.pedestrian_data[closest_ped_id]['last_update_time'] >= self.data_acquisition_period :
    #                     # rospy.loginfo("Appending new point to Id: %d", closest_ped_id)
                        
    #                     self.pedestrian_data[closest_ped_id]['x'].append(x_global)
    #                     self.pedestrian_data[closest_ped_id]['y'].append(y_global)
    #                     self.pedestrian_data[closest_ped_id]['last_update_time'] = rospy.Time.now()

    def visualization_marker_callback(self, msg):
    # in visualization_marker_callback from lstm_node_array_wg_perception_people_leg_detector.py we assumed to not trust the label given by the leg detector to each pedestrian, because
    # it was unstable (it changed too much). Instead, this leg tracker seems to be more robust: once a pedestrian is detected, the leg tracker assigns a label and it doesn't change much.
    # the label is in msg.text.
        if msg.ns == "People_tracked" and msg.points:
            with self.data_lock:
                # Get the position of the center of the legs in odom frame
                x_odom = 0.5*(msg.points[0].x + msg.points[1].x)
                y_odom = 0.5*(msg.points[0].y + msg.points[1].y)

                pedestrian_id = int(msg.text)

                # Get the transform from odom frame to global frame
                try:
                    now = rospy.Time.now()
                    self.listener.waitForTransform(self.global_frame, self.odom_frame, now, rospy.Duration(1.0))
                    (trans, rot) = self.listener.lookupTransform(self.global_frame, self.odom_frame, now)
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.logerr("TF Exception occurred")
                    return

                # Create a transformation matrix from the obtained translation and rotation
                transform_matrix = tf.transformations.quaternion_matrix(rot)
                transform_matrix[0:3, 3] = trans

                # Convert the odom coordinates to homogeneous coordinates
                point_local = np.array([x_odom, y_odom, 0, 1])

                # Transform the point from the odom frame to the global frame
                point_global = np.dot(transform_matrix, point_local)
                x_global, y_global = point_global[0], point_global[1]

                #----------------------------------------------------
                # new pedestrian detected
                if pedestrian_id not in self.pedestrian_data.keys():
                    # rospy.loginfo("New ped detected. Id: %d", pedestrian_id)
                    self.pedestrian_data[pedestrian_id] = {
                            'x': deque(maxlen=self.obs_length),
                            'y': deque(maxlen=self.obs_length),
                            'last_update_time': rospy.Time.now()
                    }
                    # rospy.loginfo("new ped detected(%d) Pedestrians ids:", pedestrian_id)
                    # for key in self.pedestrian_data.keys():
                    #     rospy.loginfo(key)
                    while len(self.pedestrian_data[pedestrian_id]['x']) < self.obs_length:
                        self.pedestrian_data[pedestrian_id]['x'].append(x_global)
                        self.pedestrian_data[pedestrian_id]['y'].append(y_global)
                        self.pedestrian_data[pedestrian_id]['last_update_time'] = rospy.Time.now()
                # pedestrian already detected => append new coordinates to the fixed-size moving window every self.data_acquisition_period seconds
                elif rospy.Time.now() - self.pedestrian_data[pedestrian_id]['last_update_time'] >= self.data_acquisition_period:
                    self.pedestrian_data[pedestrian_id]['x'].append(x_global)
                    self.pedestrian_data[pedestrian_id]['y'].append(y_global)
                    self.pedestrian_data[pedestrian_id]['last_update_time'] = rospy.Time.now()


                #----------------------------------------------------
                # # pedestrian_data is void: start filling the first position with data coming from the msg
                # if not self.pedestrian_data:
                #     rospy.loginfo("One ped detected. Id: %d", pedestrian_id)
                #     self.pedestrian_data[pedestrian_id] = {
                #             'x': deque(maxlen=self.obs_length),
                #             'y': deque(maxlen=self.obs_length),
                #             # 'window_full': False,
                #             'last_update_time': rospy.Time.now()
                #         }
                #     while len(self.pedestrian_data[pedestrian_id]['x']) < self.obs_length:
                #         self.pedestrian_data[0]['x'].append(x_global)
                #         self.pedestrian_data[0]['y'].append(y_global)
                #         # self.pedestrian_data[0]['window_full'] = True
                #         self.pedestrian_data[0]['last_update_time']: rospy.Time.now()

                # else:
                #     closest_ped_id, min_distance = self.find_closest_pedestrian(x_global, y_global)

                #     # A new pedestrian is detected: collect data as fast as possible, until the window is full, to start the prediction.
                #     if min_distance > self.intimate_distance_threshold:
                #         first_free_id = self.first_free_id()
                #         rospy.loginfo("New ped detected. Id: %d", first_free_id)
                #         self.pedestrian_data[first_free_id] = {
                #             'x': deque(maxlen=self.obs_length),
                #             'y': deque(maxlen=self.obs_length),
                #             'last_update_time': rospy.Time.now()
                #         }
                #         while len(self.pedestrian_data[first_free_id]['x']) < self.obs_length:
                #             self.pedestrian_data[first_free_id]['x'].append(x_global)
                #             self.pedestrian_data[first_free_id]['y'].append(y_global)
                #         self.pedestrian_data[first_free_id]['last_update_time'] = rospy.Time.now()
                        
                #         rospy.loginfo("Current pedestrian IDs: %s", list(self.pedestrian_data.keys()))


                #     # Append the new point to trajectory of closest_ped_id every self.data_acquisition_period seconds
                #     elif min_distance <= self.intimate_distance_threshold and rospy.Time.now() - self.pedestrian_data[closest_ped_id]['last_update_time'] >= self.data_acquisition_period :
                #         # rospy.loginfo("Appending new point to Id: %d", closest_ped_id)
                        
                #         self.pedestrian_data[closest_ped_id]['x'].append(x_global)
                #         self.pedestrian_data[closest_ped_id]['y'].append(y_global)
                #         self.pedestrian_data[closest_ped_id]['last_update_time'] = rospy.Time.now()

    # def find_closest_pedestrian(self, x_global, y_global):
    #     """Find the pedestrian_id in self.pedestrian_data with the minimum distance to (x_global, y_global)."""
        
    #     min_distance = float('inf')
    #     closest_pedestrian_id = None

    #     # Iterate through all pedestrians to find the closest one
    #     for pedestrian_id, data in self.pedestrian_data.items():
    #         if data['x'] and data['y']:  # Ensure the lists are not empty
    #             last_x = data['x'][-1]
    #             last_y = data['y'][-1]

    #             # Calculate the Euclidean distance
    #             distance = np.sqrt((x_global - last_x) ** 2 + (y_global - last_y) ** 2)

    #             # Update the minimum distance and closest pedestrian_id if this distance is smaller
    #             if distance < min_distance:
    #                 min_distance = distance
    #                 closest_pedestrian_id = pedestrian_id

    #     return closest_pedestrian_id, min_distance

    # def first_free_id(self):
    #     """Finds the first free pedestrian ID."""
    #     if not self.pedestrian_data:
    #         return 0

    #     # Sort the existing pedestrian IDs
    #     existing_ids = sorted(self.pedestrian_data.keys())

    #     # Find the first missing ID
    #     for i in range(len(existing_ids)):
    #         if existing_ids[i] != i:
    #             return i

    #     # If all IDs are sequential, return the next available ID
    #     return len(existing_ids)

    def shuffle(self, paths):
        """
        Shuffles the paths so that each pedestrian becomes the primary pedestrian.
        """
        num_pedestrians = len(paths)
        shuffled_paths_list = []

        # Generate shuffled paths where each pedestrian is the primary one
        for i in range(num_pedestrians):
            shuffled_paths = paths[i:] + paths[:i]  # Rotate the list so each pedestrian becomes the first one
            shuffled_paths_list.append(shuffled_paths)
        
        return shuffled_paths_list
        # return paths

    def xy_to_paths_mod(self, xy_paths, ped_id):
        return [trajnetplusplustools.TrackRow(i, ped_id, xy_paths[i, 0].item(), xy_paths[i, 1].item(), 0, 0)
                for i in range(len(xy_paths))]

    # def process_data_original(self):
    #     """Thread function to handle prediction and publishing."""
    #     while not rospy.is_shutdown():
    #         time.sleep(self.thread_period)  # run at 1/self.thread_period Hz
            
    #         with self.data_lock:

    #             # remove outdated pedestrians data
    #             pedestrians_to_remove = []
    #             for pedestrian_id in self.pedestrian_data:
    #                 if(rospy.Time.now () - self.pedestrian_data[pedestrian_id]['last_update_time'] >= self.deprecated_data_time):
    #                     pedestrians_to_remove.append(pedestrian_id)
    #             for pedestrian_id in pedestrians_to_remove:
    #                 rospy.loginfo("Deleting ped %d (deprecated)", pedestrian_id)
    #                 del self.pedestrian_data[pedestrian_id]
    #             # rospy.loginfo("Current pedestrian IDs: %s", list(self.pedestrian_data.keys()))

    #             # Initialize the PathArray message
    #             path_array_msg = PathArray()
    #             path_array_msg.header.stamp = rospy.Time.now()
    #             path_array_msg.header.frame_id = self.global_frame
    #             path_array_msg.pedestrian_id = []  # Array of pedestrian IDs
    #             path_array_msg.path = []  # Array of paths

    #             # LSTM prediction
    #             for pedestrian_id, data in self.pedestrian_data.items():
    #                 # Convert from deque to NumPy array
    #                 combined = np.array([[data['x'][i], data['y'][i]] for i in range(self.obs_length)])
    #                 paths = trajnetbaselines.lstm.utils.xy_to_paths(combined)
    #                 paths = [paths]

    #                 scene_goals_np = np.zeros((len(paths), 2))
    #                 scene_goals = torch.tensor(scene_goals_np, dtype=torch.float)

    #                 # Feed the LSTM and get predicted trajectory
    #                 predicted_trajectory = self.predictor(paths, scene_goals, n_predict=12, obs_length=9, modes=1, args=self.args)
    #                 predicted_trajectory = predicted_trajectory[0][0]

    #                 pred_path_msg = self.prediction_to_path(predicted_trajectory)

    #                 # Create publisher for this pedestrian if not created yet
    #                 self.create_publisher(pedestrian_id)
                    
    #                 # Publish the observed path and the predicted path for this pedestrian
    #                 self.pred_traj_pubs[pedestrian_id].publish(pred_path_msg)
                
    #                 # Add the pedestrian ID and its respective path to the PathArray
    #                 path_array_msg.pedestrian_id.append(pedestrian_id)
    #                 path_array_msg.path.append(pred_path_msg)

    #             # Publish the PathArray message containing all the pedestrian IDs and paths
    #             if path_array_msg.pedestrian_id and path_array_msg.path:
    #                 self.predicted_trajectory_pub.publish(path_array_msg)

    def process_data(self):
        """Thread function to handle prediction and publishing."""
        while not rospy.is_shutdown():
            time.sleep(self.thread_period)  # run at 1/self.thread_period Hz
            
            with self.data_lock:

                # remove outdated pedestrians data
                pedestrians_to_remove = []
                for pedestrian_id in self.pedestrian_data.keys():
                    if(rospy.Time.now() - self.pedestrian_data[pedestrian_id]['last_update_time'] >= self.deprecated_data_time):
                        pedestrians_to_remove.append(pedestrian_id)
                for pedestrian_id in pedestrians_to_remove:
                    rospy.loginfo("Deleting ped %d (deprecated)", pedestrian_id)
                    del self.pedestrian_data[pedestrian_id]
                # rospy.loginfo("Current pedestrian IDs: %s", list(self.pedestrian_data.keys()))

                # Initialize the PathArray message
                path_array_msg = PathArray()
                path_array_msg.header.stamp = rospy.Time.now()
                path_array_msg.header.frame_id = self.global_frame
                path_array_msg.pedestrian_id = []  # Array of pedestrian IDs
                path_array_msg.path = []  # Array of paths

                # LSTM prediction
                #-------------------------------------------------------------------------------------------
                network_input = []
                for pedestrian_id, data in self.pedestrian_data.items():
                    # Prepare observed trajectory data to be published by self.obs_traj_pubs
                    obs_path_msg = Path()
                    obs_path_msg.header.stamp = rospy.Time.now()
                    obs_path_msg.header.frame_id = self.global_frame

                    # Populate the observed path with recent positions
                    for i in range(self.obs_length):
                        pose = PoseStamped()
                        pose.header.stamp = rospy.Time.now()
                        pose.header.frame_id = self.global_frame
                        pose.pose.position.x = data['x'][i]
                        pose.pose.position.y = data['y'][i]
                        pose.pose.position.z = 0
                        obs_path_msg.poses.append(pose)
                    
                    self.create_publisher(pedestrian_id)
                    self.obs_traj_pubs[pedestrian_id].publish(obs_path_msg)
                    
                    # Prepare the input for LSTM prediction
                    ped_traj = np.array([[data['x'][i], data['y'][i]] for i in range(self.obs_length)])
                    ped_traj_trackrow = self.xy_to_paths_mod(ped_traj, pedestrian_id)
                    network_input.append(ped_traj_trackrow)

                # Shuffle the network input so that each pedestrian gets its prediction
                shuffled_network_input = self.shuffle(network_input)

                # Predict the trajectory for the primary pedestrian (a new pedestrian is considered the primary pedestrian at each iteration) 
                for path in shuffled_network_input:
                    scene_goals_np = np.zeros((len(path), 2))
                    scene_goals = torch.tensor(scene_goals_np, dtype=torch.float)
                    prediction = self.predictor(path, scene_goals, n_predict=12, obs_length=9, modes=1, args=self.args)
                    prediction = prediction[0][0]
                    # prediction = prediction[0:5] # limit the number of predicted trajectory

                    pred_path_msg = self.prediction_to_path(prediction)

                    # Create publisher for this pedestrian if not created yet
                    pedestrian_id = path[0][0].pedestrian
                    self.create_publisher(pedestrian_id)
                    
                    # Publish the observed path and the predicted path for this pedestrian
                    self.pred_traj_pubs[pedestrian_id].publish(pred_path_msg)
                    # self.obs_traj_pubs[pedestrian_id].publish(obs_path_msg)
                
                    # Add the pedestrian ID and its respective path to the PathArray
                    path_array_msg.pedestrian_id.append(pedestrian_id)
                    path_array_msg.path.append(pred_path_msg)
                #-------------------------------------------------------------------------------------------
                
                # CollisionDetection per condizionare la pubblicazione della traiettoria predetta

                # Publish the PathArray message containing all the pedestrian IDs and paths
                if path_array_msg.pedestrian_id and path_array_msg.path:
                    self.path_array_msg = path_array_msg
                    self.predicted_trajectory_pub.publish(path_array_msg)

    # Collision Detection -------------------------------------------------------------------------------------------------------
    def people_callback(self, msg):
        for person in msg.people:
            pedestrian_id = person.name
            self.people_velocities[pedestrian_id] = [person.velocity.x, person.velocity.y, person.velocity.z]
        self.compute_predicted_trajectory_timestamps()

    def compute_predicted_trajectory_timestamps(self):
        with self.data_lock:
            for pedestrian_id in self.pedestrian_data:
                current_position_x = self.pedestrian_data[pedestrian_id]['x'][-1]
                current_position_y = self.pedestrian_data[pedestrian_id]['y'][-1]
                # current_position = np.array([current_position_x, current_position_y])
                # rospy.loginfo(current_position)
    # --------------------------------------------------------------------------------------------------------------------------


class CollisionDetection:
    def __init__(self, lstm):
        self.lstm = lstm

        self.people_sub = rospy.Subscriber('/people', People, self.people_callback)
    #     self.lstm_node_array_sub = rospy.Subscriber('/predicted_trajectory', PathArray, self.lstm_node_array_callback)

    #     self.lstm_output_kalman_timestamps_pub_ = rospy.Publisher('/lstm_output_kalman_timestamps', Float64MultiArray, queue_size=10)

        self.people_velocities = {}
    #     self.people_current_positions = {}
    #     self.predicted_paths = {}


    def people_callback(self, msg):
        with self.lstm.data_lock:
            # rospy.loginfo(self.lstm.path_array_msg)
            for person in msg.people:
                person_id = person.name
    # #         self.people_current_positions[person_id] = [person.position.x, person.position.y, person.position.z]
                self.people_velocities[person_id] = [person.velocity.x, person.velocity.y, person.velocity.z]
            self.compute_predicted_trajectory_timestamps()


    # def lstm_node_array_callback(self, msg):

    #     # remove outdated pedestrians data
    #     pedestrians_to_remove = []
    #     for pedestrian_id in self.predicted_paths:
    #         if(rospy.Time.now () - self.pedestrian_data[pedestrian_id]['last_update_time'] >= self.deprecated_data_time):
    #             pedestrians_to_remove.append(pedestrian_id)
    #     for pedestrian_id in pedestrians_to_remove:
    #         rospy.loginfo("Deleting ped %d (deprecated)", pedestrian_id)
    #         del self.pedestrian_data[pedestrian_id]
    #     for path in msg.path:
    #         self.predicted_paths[msg.pedestrian_id] = path

    #     print(self.people_velocities)
    #     # print(self.people_velocities[person_id])
        
    #     # self.compute_predicted_trajectory_timestamps()


    def compute_predicted_trajectory_timestamps(self):
        for pedestrian_id in self.lstm.pedestrian_data:
            current_position_x = self.lstm.pedestrian_data[pedestrian_id]['x'][-1]
            current_position_y = self.lstm.pedestrian_data[pedestrian_id]['y'][-1]
            current_position = np.array([current_position_x, current_position_y])
            rospy.loginfo(current_position)

        # for person_id, path in 
        #     # Extract current position and velocity of the person
            # velocity = np.array(self.people_velocities.get(person_id, [0.0, 0.0, 0.0]))
        #     current_position = np.array(self.people_current_positions.get(person_id, [0.0, 0.0, 0.0]))


    #         velocity         = np.array(self.people_velocities.get(person_id))
    #         print("velocity ", velocity)

    #         # # Calculate the velocity magnitude
    #         # # velocity_magnitude = np.linalg.norm(velocity[:2])  # Use x, y components for velocity computation
    #         # velocity_magnitude = np.linalg.norm(velocity)  # Use x, y components for velocity computation

    #         # if velocity_magnitude == 0:
    #         #     rospy.logwarn(f"Person {person_id} has zero velocity; skipping timestamp computation.")
    #         #     continue

    #         # # Initialize matrix to store positions and timestamps
    #         # trajectory_data = []
    #         # previous_position = current_position[:2]  # Only x, y components for 2D
    #         # total_time = 0.0  # Cumulative time for timestamps

    #         # # Iterate through the path's poses
    #         # for pose in path.poses:
    #         #     # Get the 2D position from the pose
    #         #     current_pose_position = np.array([pose.pose.position.x, pose.pose.position.y])

    #         #     # Compute the Euclidean distance between current and previous position
    #         #     distance = np.linalg.norm(current_pose_position - previous_position)

    #         #     # Compute timestamp increment based on velocity
    #         #     time_increment = distance / velocity_magnitude
    #         #     total_time += time_increment

    #         #     # Append the 2D position and timestamp to the matrix
    #         #     trajectory_data.append([current_pose_position[0], current_pose_position[1], total_time])

    #         #     # Update the previous position
    #         #     previous_position = current_pose_position

    #         # # Publish the trajectory data
    #         # self.publish_trajectory_data(person_id, trajectory_data)
    
    
    # def publish_trajectory_data(self, person_id, trajectory_data):
    #     # Create a Float64MultiArray message
    #     msg = Float64MultiArray()

    #     # Set the layout (dimensions of the matrix)
    #     msg.layout.dim = [
    #         MultiArrayDimension(label='rows', size=len(trajectory_data), stride=3 * len(trajectory_data)),
    #         MultiArrayDimension(label='columns', size=3, stride=3)
    #     ]
    #     msg.layout.data_offset = 0

    #     # Flatten the trajectory data matrix into a single list
    #     flat_data = [item for sublist in trajectory_data for item in sublist]
    #     msg.data = flat_data

    #     # Log and publish the message
    #     rospy.loginfo(f"Publishing trajectory data for person {person_id}: {msg}")
    #     self.lstm_output_kalman_timestamps_pub_.publish(msg)


def add_arguments(parser):
    parser.add_argument('--path', default='DATA_BLOCK/synth_data/test_pred/',
                        help='directory of data to test')
    parser.add_argument('--output', nargs='+',
                        help='relative path to saved model')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--write_only', action='store_true',
                        help='disable writing new files')
    parser.add_argument('--disable-collision', action='store_true',
                        help='disable collision metrics')
    parser.add_argument('--labels', required=False, nargs='+',
                        help='labels of models')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='augment scenes')
    parser.add_argument('--modes', default=1, type=int,
                        help='number of modes to predict')
    parser.add_argument('--scene_id', default=1888, type=int,
                        help='scene id')
    parser.add_argument('--neural_network_model', type=str, default="",
                        help='Type of the model (e.g., lstm, sgan, vae)')
    parser.add_argument('--interaction_module', type=str, default="",
                        help='Interaction module (e.g., vanilla, occupancy, directional, social, hiddenstatemlp, nn, attentionmlp, nn_lstm, traj_pool)')

    return parser.parse_known_args()


def main():
    rospy.init_node('lstm_node_array3')
    lstm = LstmNodeArray()
    # collision_detection = CollisionDetection(lstm)
    rospy.spin()


if __name__ == '__main__':
    main()