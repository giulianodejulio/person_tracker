#!/usr/bin/env python3

import rospy
import rospkg # to load ROS package paths dynamically
import numpy as np
from collections import deque # to have fixed-size moving window
import argparse  # to define some default args for self.predictor() to work

from visualization_msgs.msg import MarkerArray # to handle /legs_markers message type
from visualization_msgs.msg import Marker # to display predicted trajectory in RViz
from geometry_msgs.msg import Point  # to display predicted trajectory in RViz
# from geometry_msgs.msg import PointStamped  # format of output trajectory
from nav_msgs.msg import Path # format of output trajectory

from geometry_msgs.msg import PoseStamped

import tf

# LSTM imports
# import trajnetplusplustools
import trajnetbaselines.lstm.trajnet_evaluator as te # to use LSTM model from Trajnet++
import trajnetbaselines.lstm.utils
import torch # to use tensors


class LstmNode:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args, _ = add_arguments(self.parser) # use this instead of self.args = add_arguments(self.parser) and return parser.parse_known_args() because it is how i make it work when launching lstm_node.py in the mobile_base_autonomous_navigation.launch 

        ## APPROACH 1: Subsampling of data coming from /legs_markers (incoming data is too fast, more than 24 Hz)
        self.last_iter_completion_time = rospy.Time.now()
        rospy.logerr("self.last_iter_completion_time %f", self.last_iter_completion_time.to_sec())
        self.loop_period = rospy.Duration(1) # default value: 1 second
        ## APPROACH 2: Process every (subsampling_rate)th message (at ~1 Hz if incoming data is at subsampling_rate Hz)
        # self.counter = 0
        # self.subsampling_rate = 10
        ## APPROACH 3: Use rospy.Rate and rate.sleep()
        # loop_period_seconds = self.loop_period.to_sec()
        # self.rate = rospy.Rate(1.0/loop_period_seconds)
        # self.rate = rospy.Rate(1.0)

        self.obs_length = 9
        # Fixed-size temporal window to feed the LSTM
        self.x = deque(maxlen=self.obs_length)
        self.y = deque(maxlen=self.obs_length)
        
        # Dynamically get the model path
        package_path = rospkg.RosPack().get_path('hrii_person_tracker')
        self.model = f"{package_path}/scripts/lstm_vanilla_None.pkl"

        # Load LSTM model using Trajnet++
        self.predictor  = te.load_predictor(self.model)

        # tf listener to get transform of ($ arg robot_id)_front_laser_link wrt ($ arg robot_id)_map
        self.mobile_base = rospy.get_param("~mobile_base")
        self.global_frame = self.mobile_base + "_map"              # e.g.: 'rbkairos_map'
        self.laser_frame = self.mobile_base + "_front_laser_link"  # e.g.: 'rbkairos_front_laser_link'
        self.listener = tf.TransformListener()

        self.legs_markers_sub = rospy.Subscriber('/legs_markers', MarkerArray, self.legs_markers_callback)
        self.pred_traj_pub = rospy.Publisher('/predicted_trajectory', Path, queue_size=10)
        # self.pred_traj_pub = rospy.Publisher('/predicted_trajectory', PointStamped, queue_size=10)

    ## Publish the predicted trajectory as PointStamped messages
    # def publish_points(self, points):
        # for i, point in enumerate(points):
        #     point_msg = PointStamped()
        #     point_msg.header.frame_id = self.laser_frame
        #     point_msg.header.stamp = rospy.Time.now()
        #     point_msg.point.x = point[0]
        #     point_msg.point.y = point[1]
        #     point_msg.point.z = 0  # 2D prediction => z=0
        #     self.pred_traj_pub.publish(point_msg)

    def prediction_to_path(self, predicted_trajectory):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.global_frame
        # path.header.frame_id = self.laser_frame
        for point in predicted_trajectory:
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = self.global_frame
            # pose.header.frame_id = self.laser_frame
            pose.pose.position.x = point[0] # 1.0
            pose.pose.position.y = point[1] # 0.0
            pose.pose.position.z = 0  # 2D prediction => z=0
            pose.pose.orientation.w = 1.0  # no orientation considered
            path.poses.append(pose)
        return path

    
    def legs_markers_callback(self, msg):
        ## APPROACH 1
        ## Receive data from /legs_markers at a lower frequency
        if rospy.Time.now() - self.last_iter_completion_time < self.loop_period:
            return
        rospy.logerr("len(self.x) %d after %f seconds", len(self.x), rospy.Time.now().to_sec())
        
        ## APPROACH 2
        # self.counter += 1
        # if self.counter < self.subsampling_rate:
        #     return
        # self.counter = 0
        ## APPROACH 3: using rate.sleep() at the end of the loop
        # self.rate = rospy.Rate(1.0)


        x0 = msg.markers[0].pose.position.x
        y0 = msg.markers[0].pose.position.y
        x1 = msg.markers[1].pose.position.x
        y1 = msg.markers[1].pose.position.y

        ## Transform (x0,y0) and (x1,y1) from self.laser_frame to self.global_frame
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform(self.global_frame, self.laser_frame, now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform(self.global_frame, self.laser_frame, now)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("TF Exception occurred")
            return
        
        # Create a transformation matrix from the obtained translation and rotation
        transform_matrix = tf.transformations.quaternion_matrix(rot)
        transform_matrix[0:3, 3] = trans

        # Convert (x0, y0) and (x1, y1) to homogeneous coordinates
        point0_local = np.array([x0, y0, 0, 1])
        point1_local = np.array([x1, y1, 0, 1])

        # Transform points from local (sensor) frame to global frame
        point0_global = np.dot(transform_matrix, point0_local)
        point1_global = np.dot(transform_matrix, point1_local)

        x0_global, y0_global = point0_global[0], point0_global[1]
        x1_global, y1_global = point1_global[0], point1_global[1]

        # Compute the centroid in the global frame
        x_centroid = (x0_global + x1_global) / 2
        y_centroid = (y0_global + y1_global) / 2

        self.x.append(x_centroid)
        self.y.append(y_centroid)

        if len(self.x) == self.obs_length and len(self.y) == self.obs_length: # start predicting only after enough points have been observed
            ## Convert from deque to Trajnet's TrackRow type to use __call__ method of LSTMPredictor class
            ## Combine self.x and self.y into a NumPy array
            combined = np.array([[self.x[i], self.y[i]] for i in range(self.obs_length)])
            paths = trajnetbaselines.lstm.utils.xy_to_paths(combined)
            paths = [paths] # this is the format output by trajnetplusplustools.Reader.paths_to_xy(paths) which is in LSTMPredictor.__call__()


            ###### scene_goals = [np.zeros((len(paths), 2))] # THIS IS INEFFICIENT:
            ###### rosrun hrii_person_tracker lstm_node.py 
            ###### /root/ros_ws/src/hrii_person_tracker/trajnetpp/trajnetplusplusbaselines/trajnetbaselines/lstm/lstm.py:297: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at  ../torch/csrc/utils/tensor_new.cpp:201.)
            ###### scene_goal = torch.Tensor(scene_goal) #.to(device)

            # Instead I can do:
            scene_goals_np = np.zeros((len(paths), 2))
            scene_goals = torch.tensor(scene_goals_np, dtype=torch.float)


            ## Feed the LSTM and get predicted trajectory
            predicted_trajectory = self.predictor(paths, scene_goals, n_predict=12, obs_length=9, modes=1, args=self.args)
            predicted_trajectory = predicted_trajectory[0][0]
            # rospy.loginfo(f"predictions: {predicted_trajectory}")


            ## Convert to nav_msgs::Path message and publish
            path_msg = self.prediction_to_path(predicted_trajectory)
            # Publish the Path message
            # rate = rospy.Rate(0.5)  # 0.5 Hz
            # while not rospy.is_shutdown():
            self.pred_traj_pub.publish(path_msg)
                # rate.sleep()
            
            # self.publish_points(predicted_trajectory)


            ## APPROACH 1
            self.last_iter_completion_time = rospy.Time.now()
        ## APPROACH 3
        # self.rate.sleep()

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
    rospy.init_node('lstm_node')
    lstm = LstmNode()
    rospy.spin()


if __name__ == '__main__':
    main()
