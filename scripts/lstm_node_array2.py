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
## import trajnetplusplustools
import trajnetbaselines.lstm.trajnet_evaluator as te  # to use LSTM model from Trajnet++
import trajnetbaselines.lstm.utils
import torch  # to use tensors

# dynamic ROS parameters
from dynamic_reconfigure.server import Server
from hrii_person_tracker.cfg import LstmNodeConfig

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

        self.intimate_distance_threshold = 1.0
        self.obs_length = 9

        # Store a fixed-size moving window for each pedestrian in a dictionary of dictionaries.
        # pedestrian_data =
        # {'0': {'x': ..., 'y': ..., 'last_update_time': ...,},
        #  '1': {'x': ..., 'y': ..., 'last_update_time': ...,}
        #  ...
        # }
        self.pedestrian_data = {}
        self.data_lock = threading.Lock()
                
        # Dynamically get the model path
        package_path = rospkg.RosPack().get_path('hrii_person_tracker')
        self.model = f"{package_path}/scripts/lstm_vanilla_None.pkl"

        # Load LSTM model using Trajnet++
        self.predictor = te.load_predictor(self.model)

        # tf listener to get transform of odom frame rbkairos_odom wrt global frame rbkairos_map (because leg_detector works with odom frame)
        self.mobile_base = rospy.get_param("~mobile_base")
        self.global_frame = self.mobile_base + "_map"
        self.odom_frame = self.mobile_base + "_odom"
        self.listener = tf.TransformListener()

        # Subscriber for visualization markers
        self.visualization_marker_sub = rospy.Subscriber('/visualization_marker', Marker, self.visualization_markers_callback)

        # Dynamic publishers for each pedestrian based on their ID
        self.pred_traj_pubs = {}

        #----------------------------------------------------------------
        # Create a single publisher for PathArray
        self.predicted_trajectory_pub = rospy.Publisher('/predicted_trajectory', PathArray, queue_size=10)
        #----------------------------------------------------------------

        # Start prediction thread
        self.prediction_thread = threading.Thread(target=self.process_data)
        self.prediction_thread.start()

    def create_publisher(self, pedestrian_id):
        """Create a publisher for a pedestrian's predicted trajectory if it does not exist."""
        if pedestrian_id not in self.pred_traj_pubs:
            topic_name = f'/predicted_trajectory{pedestrian_id}'
            self.pred_traj_pubs[pedestrian_id] = rospy.Publisher(topic_name, Path, queue_size=10)
            rospy.loginfo(f"Created pub {pedestrian_id}")

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

    def visualization_markers_callback(self, msg):

        with self.data_lock:
            # Get the position of the marker in odom frame
            x_odom = msg.pose.position.x
            y_odom = msg.pose.position.y

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

            # pedestrian_data is void: start filling the first position with data coming from the msg
            if not self.pedestrian_data:
                rospy.loginfo("One ped detected. Id: 0")
                self.pedestrian_data[0] = {
                        'x': deque(maxlen=self.obs_length),
                        'y': deque(maxlen=self.obs_length),
                        # 'window_full': False,
                        'last_update_time': rospy.Time.now()
                    }
                while len(self.pedestrian_data[0]['x']) < self.obs_length:
                    self.pedestrian_data[0]['x'].append(x_global)
                    self.pedestrian_data[0]['y'].append(y_global)
                    # self.pedestrian_data[0]['window_full'] = True
                    self.pedestrian_data[0]['last_update_time']: rospy.Time.now()

            else:
                closest_ped_id, min_distance = self.find_closest_pedestrian(x_global, y_global)

                # A new pedestrian is detected: collect data as fast as possible, until the window is full, to start the prediction.
                if min_distance > self.intimate_distance_threshold:
                    first_free_id = self.first_free_id()
                    rospy.loginfo("New ped detected. Id: %d", first_free_id)
                    self.pedestrian_data[first_free_id] = {
                        'x': deque(maxlen=self.obs_length),
                        'y': deque(maxlen=self.obs_length),
                        'last_update_time': rospy.Time.now()
                    }
                    while len(self.pedestrian_data[first_free_id]['x']) < self.obs_length:
                        self.pedestrian_data[first_free_id]['x'].append(x_global)
                        self.pedestrian_data[first_free_id]['y'].append(y_global)
                    self.pedestrian_data[first_free_id]['last_update_time'] = rospy.Time.now()
                    
                    rospy.loginfo("Current pedestrian IDs: %s", list(self.pedestrian_data.keys()))


                # Append the new point to trajectory of closest_ped_id every self.data_acquisition_period seconds
                elif min_distance <= self.intimate_distance_threshold and rospy.Time.now() - self.pedestrian_data[closest_ped_id]['last_update_time'] >= self.data_acquisition_period :
                    # rospy.loginfo("Appending new point to Id: %d", closest_ped_id)
                    
                    self.pedestrian_data[closest_ped_id]['x'].append(x_global)
                    self.pedestrian_data[closest_ped_id]['y'].append(y_global)
                    self.pedestrian_data[closest_ped_id]['last_update_time'] = rospy.Time.now()

    def find_closest_pedestrian(self, x_global, y_global):
        """Find the pedestrian_id in self.pedestrian_data with the minimum distance to (x_global, y_global)."""
        
        min_distance = float('inf')
        closest_pedestrian_id = None

        # Iterate through all pedestrians to find the closest one
        for pedestrian_id, data in self.pedestrian_data.items():
            if data['x'] and data['y']:  # Ensure the lists are not empty
                last_x = data['x'][-1]
                last_y = data['y'][-1]

                # Calculate the Euclidean distance
                distance = np.sqrt((x_global - last_x) ** 2 + (y_global - last_y) ** 2)

                # Update the minimum distance and closest pedestrian_id if this distance is smaller
                if distance < min_distance:
                    min_distance = distance
                    closest_pedestrian_id = pedestrian_id

        return closest_pedestrian_id, min_distance

    def first_free_id(self):
        """Finds the first free pedestrian ID."""
        if not self.pedestrian_data:
            return 0

        # Sort the existing pedestrian IDs
        existing_ids = sorted(self.pedestrian_data.keys())

        # Find the first missing ID
        for i in range(len(existing_ids)):
            if existing_ids[i] != i:
                return i

        # If all IDs are sequential, return the next available ID
        return len(existing_ids)

    def process_data(self):
        """Thread function to handle prediction and publishing."""
        while not rospy.is_shutdown():
            time.sleep(self.thread_period)  # run at 1/self.thread_period Hz
            
            with self.data_lock:

                # remove outdated pedestrians data
                pedestrians_to_remove = []
                for pedestrian_id in self.pedestrian_data:
                    if(rospy.Time.now () - self.pedestrian_data[pedestrian_id]['last_update_time'] >= self.deprecated_data_time):
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
                for pedestrian_id, data in self.pedestrian_data.items():
                    # Convert from deque to NumPy array
                    combined = np.array([[data['x'][i], data['y'][i]] for i in range(self.obs_length)])
                    paths = trajnetbaselines.lstm.utils.xy_to_paths(combined)
                    paths = [paths]

                    scene_goals_np = np.zeros((len(paths), 2))
                    scene_goals = torch.tensor(scene_goals_np, dtype=torch.float)

                    # Feed the LSTM and get predicted trajectory
                    predicted_trajectory = self.predictor(paths, scene_goals, n_predict=12, obs_length=9, modes=1, args=self.args)
                    predicted_trajectory = predicted_trajectory[0][0]

                    path_msg = self.prediction_to_path(predicted_trajectory)

                    # Create publisher for this pedestrian if not created yet
                    self.create_publisher(pedestrian_id)
                    
                    # Publish the predicted path for this pedestrian
                    self.pred_traj_pubs[pedestrian_id].publish(path_msg)
                
                    # Add the pedestrian ID and its respective path to the PathArray
                    path_array_msg.pedestrian_id.append(pedestrian_id)
                    path_array_msg.path.append(path_msg)

                # Publish the PathArray message containing all the pedestrian IDs and paths
                if path_array_msg.pedestrian_id and path_array_msg.path:
                    self.predicted_trajectory_pub.publish(path_array_msg)

    # def visualization_markers_callback(self, msg):
    #     """Callback to handle incoming markers and collect data."""
    #     if msg.ns == "PEOPLE":
    #         pedestrian_id = msg.id

    #         # Initialize the fixed-size moving window for the pedestrian if it does not exist
    #         with self.data_lock:
    #             if pedestrian_id not in self.pedestrian_data:
    #                 self.pedestrian_data[pedestrian_id] = {
    #                     'x': deque(maxlen=self.obs_length),
    #                     'y': deque(maxlen=self.obs_length),
    #                     'window_full': False,
    #                     'last_update_time': rospy.Time.now()
    #                 }

               

    #             if self.pedestrian_data[pedestrian_id]['window_full']:
    #                 # check that the recorded point is not too far from the last recorded point to avoid jumps in the observed trajectory
    #                 distance = np.sqrt((x_global - self.pedestrian_data[pedestrian_id]['x'][-1]) ** 2 + (y_global - self.pedestrian_data[pedestrian_id]['y'][-1]) ** 2)
    #                 # Collect data at 1 Hz when the window is full
    #                 if distance <= 2.5 and rospy.Time.now() - self.pedestrian_data[pedestrian_id]['last_update_time'] >= self.data_acquisition_period:
    #                     # rospy.logerr("\ncollecting data at 1 Hz for ped %d", pedestrian_id)

    #                     self.pedestrian_data[pedestrian_id]['x'].append(x_global)
    #                     self.pedestrian_data[pedestrian_id]['y'].append(y_global)
    #                     self.pedestrian_data[pedestrian_id]['last_update_time'] = rospy.Time.now()

    #             else: # A new pedestrian is detected: collect data as fast as possible, until the window is full, to start the prediction soon

    #                 # rospy.logerr("collecting data fast for ped %d", pedestrian_id)
    #                 # self.pedestrian_data[pedestrian_id]['x'].append(x_global)
    #                 # self.pedestrian_data[pedestrian_id]['y'].append(y_global)
    #                 # if len(self.pedestrian_data[pedestrian_id]['x']) == self.obs_length:
    #                 #     self.pedestrian_data[pedestrian_id]['window_full'] = True

    #                 # rospy.logerr("collecting data FAST for ped %d", pedestrian_id)
    #                 # Clear previous content
    #                 self.pedestrian_data[pedestrian_id]['x'].clear()
    #                 self.pedestrian_data[pedestrian_id]['y'].clear()
    #                 # Fill until the lists reach obs_length
    #                 while len(self.pedestrian_data[pedestrian_id]['x']) < self.obs_length:
    #                     self.pedestrian_data[pedestrian_id]['x'].append(x_global)
    #                     self.pedestrian_data[pedestrian_id]['y'].append(y_global)
    #                 # Set the window_full flag once the lists are filled
    #                 self.pedestrian_data[pedestrian_id]['window_full'] = True
    #                 self.pedestrian_data[pedestrian_id]['last_update_time'] = rospy.Time.now()

    # def process_data(self):
    #     """Thread function to handle prediction and publishing."""
    #     while not rospy.is_shutdown():
    #         time.sleep(self.thread_period)  # run at 1/self.thread_period Hz
            
    #         with self.data_lock:
    #             #----------------------------------------------------------------
    #             # Initialize the PathArray message
    #             path_array_msg = PathArray()
    #             path_array_msg.header.stamp = rospy.Time.now()
    #             path_array_msg.header.frame_id = self.global_frame
    #             path_array_msg.pedestrian_id = []  # Array of pedestrian IDs
    #             path_array_msg.path = []  # Array of paths
    #             pedestrians_to_remove = [] # Pedestrians to remove from self.pedestrian_data
    #             #----------------------------------------------------------------

    #             for pedestrian_id, data in self.pedestrian_data.items():
    #                 if (rospy.Time.now() - data['last_update_time'] < self.deprecated_data_time):  # Check if data is not too old
    #                     if data['window_full']:
    #                         # Convert from deque to NumPy array
    #                         combined = np.array([[data['x'][i], data['y'][i]] for i in range(self.obs_length)])
    #                         paths = trajnetbaselines.lstm.utils.xy_to_paths(combined)
    #                         paths = [paths]

    #                         scene_goals_np = np.zeros((len(paths), 2))
    #                         scene_goals = torch.tensor(scene_goals_np, dtype=torch.float)

    #                         # Feed the LSTM and get predicted trajectory
    #                         predicted_trajectory = self.predictor(paths, scene_goals, n_predict=12, obs_length=9, modes=1, args=self.args)
    #                         predicted_trajectory = predicted_trajectory[0][0]

    #                         path_msg = self.prediction_to_path(predicted_trajectory)

    #                         # Create publisher for this pedestrian if not created yet
    #                         self.create_publisher(pedestrian_id)
                            
    #                         #----------------------------------------------------------------
    #                         # Add the pedestrian ID and its respective path to the PathArray
    #                         path_array_msg.pedestrian_id.append(pedestrian_id)
    #                         path_array_msg.path.append(path_msg)
    #                         #----------------------------------------------------------------

    #                         # Publish the predicted path for this pedestrian
    #                         self.pred_traj_pubs[pedestrian_id].publish(path_msg)
    #                 else:
    #                     #-- If data is outdated, stop publishing by unregistering the publisher
    #                     #-- if pedestrian_id in self.pred_traj_pubs:
    #                     #--     rospy.logerr(f"Stopped pub {pedestrian_id}")
    #                     #--     self.pred_traj_pubs[pedestrian_id].unregister()
    #                     #--     del self.pred_traj_pubs[pedestrian_id]

    #                     # If data is outdated, mark this pedestrian for removal
    #                     pedestrians_to_remove.append(pedestrian_id)

    #             # Remove outdated pedestrians and stop publishing after iterating through the dictionary
    #             for pedestrian_id in pedestrians_to_remove:
    #                 rospy.loginfo(f"Removing outdated pedestrian {pedestrian_id}")
    #                 del self.pedestrian_data[pedestrian_id]
    #                 if pedestrian_id in self.pred_traj_pubs:
    #                     self.pred_traj_pubs[pedestrian_id].unregister()
    #                     del self.pred_traj_pubs[pedestrian_id]
    #             #----------------------------------------------------------------
    #             # Publish the PathArray message containing all the pedestrian IDs and paths
    #             if path_array_msg.pedestrian_id and path_array_msg.path:  # Ensure there is data to publish
    #                 self.predicted_trajectory_pub.publish(path_array_msg)
    #             #----------------------------------------------------------------

    # def process_data_old(self):
    #     """Thread function to handle prediction and publishing."""
    #     while not rospy.is_shutdown():
    #         time.sleep(self.thread_period) # run at 1/self.thread_period Hz
            
    #         with self.data_lock:
    #             #----------------------------------------------------------------
    #             # Initialize the PathArray message
    #             path_array_msg = PathArray()
    #             path_array_msg.header.stamp = rospy.Time.now()
    #             path_array_msg.header.frame_id = self.global_frame
    #             path_array_msg.path = []
    #             #----------------------------------------------------------------

    #             for pedestrian_id, data in self.pedestrian_data.items():
    #                 if (rospy.Time.now() - data['last_update_time'] < self.deprecated_data_time): # check data is not too old
    #                     if data['window_full']:
    #                         # Convert from deque to NumPy array
    #                         combined = np.array([[data['x'][i], data['y'][i]] for i in range(self.obs_length)])
    #                         paths = trajnetbaselines.lstm.utils.xy_to_paths(combined)
    #                         paths = [paths]

    #                         scene_goals_np = np.zeros((len(paths), 2))
    #                         scene_goals = torch.tensor(scene_goals_np, dtype=torch.float)

    #                         # Feed the LSTM and get predicted trajectory
    #                         predicted_trajectory = self.predictor(paths, scene_goals, n_predict=12, obs_length=9, modes=1, args=self.args)
    #                         predicted_trajectory = predicted_trajectory[0][0]

    #                         path_msg = self.prediction_to_path(predicted_trajectory)

    #                         # Create publisher for this pedestrian if not created yet
    #                         self.create_publisher(pedestrian_id)
                            
    #                         #----------------------------------------------------------------
    #                         # Create a Path message for this pedestrian
    #                         path_array_msg.pedestrian_id = pedestrian_id
    #                         path_array_msg.path.append(path_msg)
    #                         # Publish the PathArray message only if it contains data
    #                         if path_array_msg.path:
    #                             self.predicted_trajectory_pub.publish(path_array_msg)
    #                         #----------------------------------------------------------------

    #                         # Publish the predicted path for this pedestrian
    #                         self.pred_traj_pubs[pedestrian_id].publish(path_msg)
    #                 else:
    #                     # If data is outdated, stop publishing by unregistering the publisher
    #                     if pedestrian_id in self.pred_traj_pubs:
    #                         rospy.loginfo(f"Stopped pub {pedestrian_id}")
    #                         self.pred_traj_pubs[pedestrian_id].unregister()
    #                         del self.pred_traj_pubs[pedestrian_id]


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
    return parser.parse_known_args()


def main():
    rospy.init_node('lstm_node_array2')
    lstm = LstmNodeArray()
    rospy.spin()


if __name__ == '__main__':
    main()