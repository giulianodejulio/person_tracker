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


class LstmNodeArray:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args, _ = add_arguments(self.parser)

        self.thread_period = 1.0
        self.data_acquisition_period = rospy.Duration(self.thread_period)  # default value: 1 second
        self.deprecated_data_time = rospy.Duration(self.thread_period)

        self.obs_length = 9

        # Store a fixed-size moving window for each pedestrian in a dictionary of dictionaries.
        # pedestrian_data =
        # {'0': {'x': ..., 'y': ..., 'window_full': ..., 'last_update_time': ...,},
        #  '1': {'x': ..., 'y': ..., 'window_full': ..., 'last_update_time': ...,}
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
        """Callback to handle incoming markers and collect data."""
        if msg.ns == "PEOPLE":
            pedestrian_id = msg.id

            # Initialize the fixed-size moving window for the pedestrian if it does not exist
            with self.data_lock:
                if pedestrian_id not in self.pedestrian_data:
                    self.pedestrian_data[pedestrian_id] = {
                        'x': deque(maxlen=self.obs_length),
                        'y': deque(maxlen=self.obs_length),
                        'window_full': False,
                        'last_update_time': rospy.Time.now()
                    }

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

                if self.pedestrian_data[pedestrian_id]['window_full']:
                    # check that the recorded point is not too far from the last recorded point to avoid jumps in the observed trajectory
                    distance = np.sqrt((x_global - self.pedestrian_data[pedestrian_id]['x'][-1]) ** 2 + (y_global - self.pedestrian_data[pedestrian_id]['y'][-1]) ** 2)
                    # Collect data at 1 Hz when the window is full
                    if distance <= 2.5 and rospy.Time.now() - self.pedestrian_data[pedestrian_id]['last_update_time'] >= self.data_acquisition_period:
                        # rospy.logerr("\ncollecting data at 1 Hz for ped %d", pedestrian_id)

                        self.pedestrian_data[pedestrian_id]['x'].append(x_global)
                        self.pedestrian_data[pedestrian_id]['y'].append(y_global)
                        self.pedestrian_data[pedestrian_id]['last_update_time'] = rospy.Time.now()

                else: # A new pedestrian is detected: collect data as fast as possible, until the window is full, to start the prediction soon

                    # rospy.logerr("collecting data fast for ped %d", pedestrian_id)
                    # self.pedestrian_data[pedestrian_id]['x'].append(x_global)
                    # self.pedestrian_data[pedestrian_id]['y'].append(y_global)
                    # if len(self.pedestrian_data[pedestrian_id]['x']) == self.obs_length:
                    #     self.pedestrian_data[pedestrian_id]['window_full'] = True

                    # rospy.logerr("collecting data FAST for ped %d", pedestrian_id)
                    # Clear previous content
                    self.pedestrian_data[pedestrian_id]['x'].clear()
                    self.pedestrian_data[pedestrian_id]['y'].clear()
                    # Fill until the lists reach obs_length
                    while len(self.pedestrian_data[pedestrian_id]['x']) < self.obs_length:
                        self.pedestrian_data[pedestrian_id]['x'].append(x_global)
                        self.pedestrian_data[pedestrian_id]['y'].append(y_global)
                    # Set the window_full flag once the lists are filled
                    self.pedestrian_data[pedestrian_id]['window_full'] = True
                    self.pedestrian_data[pedestrian_id]['last_update_time'] = rospy.Time.now()

    def process_data(self):
        """Thread function to handle prediction and publishing."""
        while not rospy.is_shutdown():
            time.sleep(self.thread_period)  # run at 1/self.thread_period Hz
            
            with self.data_lock:
                #----------------------------------------------------------------
                # Initialize the PathArray message
                path_array_msg = PathArray()
                path_array_msg.header.stamp = rospy.Time.now()
                path_array_msg.header.frame_id = self.global_frame
                path_array_msg.pedestrian_id = []  # Array of pedestrian IDs
                path_array_msg.path = []  # Array of paths
                pedestrians_to_remove = [] # Pedestrians to remove from self.pedestrian_data
                #----------------------------------------------------------------

                for pedestrian_id, data in self.pedestrian_data.items():
                    if (rospy.Time.now() - data['last_update_time'] < self.deprecated_data_time):  # Check if data is not too old
                        if data['window_full']:
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
                            
                            #----------------------------------------------------------------
                            # Add the pedestrian ID and its respective path to the PathArray
                            path_array_msg.pedestrian_id.append(pedestrian_id)
                            path_array_msg.path.append(path_msg)
                            #----------------------------------------------------------------

                            # Publish the predicted path for this pedestrian
                            self.pred_traj_pubs[pedestrian_id].publish(path_msg)
                    else:
                        #-- If data is outdated, stop publishing by unregistering the publisher
                        #-- if pedestrian_id in self.pred_traj_pubs:
                        #--     rospy.logerr(f"Stopped pub {pedestrian_id}")
                        #--     self.pred_traj_pubs[pedestrian_id].unregister()
                        #--     del self.pred_traj_pubs[pedestrian_id]

                        # If data is outdated, mark this pedestrian for removal
                        pedestrians_to_remove.append(pedestrian_id)

                # Remove outdated pedestrians and stop publishing after iterating through the dictionary
                for pedestrian_id in pedestrians_to_remove:
                    rospy.loginfo(f"Removing outdated pedestrian {pedestrian_id}")
                    del self.pedestrian_data[pedestrian_id]
                    if pedestrian_id in self.pred_traj_pubs:
                        self.pred_traj_pubs[pedestrian_id].unregister()
                        del self.pred_traj_pubs[pedestrian_id]
                #----------------------------------------------------------------
                # Publish the PathArray message containing all the pedestrian IDs and paths
                if path_array_msg.pedestrian_id and path_array_msg.path:  # Ensure there is data to publish
                    self.predicted_trajectory_pub.publish(path_array_msg)
                #----------------------------------------------------------------

    def process_data_old(self):
        """Thread function to handle prediction and publishing."""
        while not rospy.is_shutdown():
            time.sleep(self.thread_period) # run at 1/self.thread_period Hz
            
            with self.data_lock:
                #----------------------------------------------------------------
                # Initialize the PathArray message
                path_array_msg = PathArray()
                path_array_msg.header.stamp = rospy.Time.now()
                path_array_msg.header.frame_id = self.global_frame
                path_array_msg.path = []
                #----------------------------------------------------------------

                for pedestrian_id, data in self.pedestrian_data.items():
                    if (rospy.Time.now() - data['last_update_time'] < self.deprecated_data_time): # check data is not too old
                        if data['window_full']:
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
                            
                            #----------------------------------------------------------------
                            # Create a Path message for this pedestrian
                            path_array_msg.pedestrian_id = pedestrian_id
                            path_array_msg.path.append(path_msg)
                            # Publish the PathArray message only if it contains data
                            if path_array_msg.path:
                                self.predicted_trajectory_pub.publish(path_array_msg)
                            #----------------------------------------------------------------

                            # Publish the predicted path for this pedestrian
                            self.pred_traj_pubs[pedestrian_id].publish(path_msg)
                    else:
                        # If data is outdated, stop publishing by unregistering the publisher
                        if pedestrian_id in self.pred_traj_pubs:
                            rospy.loginfo(f"Stopped pub {pedestrian_id}")
                            self.pred_traj_pubs[pedestrian_id].unregister()
                            del self.pred_traj_pubs[pedestrian_id]


        # def visualization_markers_callback(self, msg):
        #     """Callback to handle incoming markers and predict trajectories."""
        #     if (rospy.Time.now() - self.last_iter_completion_time < self.data_acquisition_period): #and not(self.observed_window_is_full):
        #         return

        #     # loop_init_time = rospy.Time.now()

        #     if msg.ns == "PEOPLE":
        #         pedestrian_id = msg.id

        #         # Initialize the fixed-size moving window for the pedestrian if it does not exist
        #         if pedestrian_id not in self.x:
        #             self.x[pedestrian_id] = deque(maxlen=self.obs_length)
        #             self.y[pedestrian_id] = deque(maxlen=self.obs_length)
                
        #         rospy.logerr("len(self.x[%d]) %d, len(self.y[%d]) %d", pedestrian_id, len(self.x[pedestrian_id]), pedestrian_id, len(self.y[pedestrian_id]))

        #         # Get the position of the marker in odom_frame
        #         x_odom = msg.pose.position.x
        #         y_odom = msg.pose.position.y

        #         # Get the transform from odom frame to global frame
        #         try:
        #             now = rospy.Time.now()
        #             self.listener.waitForTransform(self.global_frame, self.odom_frame, now, rospy.Duration(1.0))
        #             (trans, rot) = self.listener.lookupTransform(self.global_frame, self.odom_frame, now)
        #         except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #             rospy.logerr("TF Exception occurred")
        #             return

        #         # Create a transformation matrix from the obtained translation and rotation
        #         transform_matrix = tf.transformations.quaternion_matrix(rot)
        #         transform_matrix[0:3, 3] = trans

        #         # Convert the odom coordinates to homogeneous coordinates
        #         point_local = np.array([x_odom, y_odom, 0, 1])

        #         # Transform the point from the odom frame to the global frame
        #         point_global = np.dot(transform_matrix, point_local)

        #         x_global, y_global = point_global[0], point_global[1]

        #         # Check if there are already points in the deque
        #         if len(self.x[pedestrian_id]) > 0:
        #             # Get the last recorded point in the global frame
        #             x_last = self.x[pedestrian_id][-1]
        #             y_last = self.y[pedestrian_id][-1]

        #             # Calculate the distance between the msg point (converted in global frame) and the last recorded point
        #             distance = np.sqrt((x_global - x_last) ** 2 + (y_global - y_last) ** 2)

        #             # If the distance is greater than 1 meter, reset the deque
        #             if distance > 2.0:
        #                 rospy.logwarn(f"Pedestrian {pedestrian_id} moved more than 1 meter. Resetting trajectory.")
                        
        #                 # Fill both x and y deque with the same point to ensure prediction can still be made (it will be just a first prediction, later refined)
        #                 self.x[pedestrian_id] = deque([x_global] * self.obs_length, maxlen=self.obs_length)
        #                 self.y[pedestrian_id] = deque([y_global] * self.obs_length, maxlen=self.obs_length)
        #                 # self.observed_window_is_full = False


        #         # Add the transformed points to the pedestrian's trajectory buffer
        #         self.x[pedestrian_id].append(x_global)
        #         self.y[pedestrian_id].append(y_global)


        #         # Predict and publish trajectory if we have enough points for the pedestrian
        #         if len(self.x[pedestrian_id]) == self.obs_length:
        #             # self.observed_window_is_full = True

        #             ## Convert from deque to NumPy array
        #             combined = np.array([[self.x[pedestrian_id][i], self.y[pedestrian_id][i]] for i in range(self.obs_length)])
        #             paths = trajnetbaselines.lstm.utils.xy_to_paths(combined)
        #             paths = [paths]
                    
        #             scene_goals_np = np.zeros((len(paths), 2))
        #             scene_goals = torch.tensor(scene_goals_np, dtype=torch.float)

        #             ## Feed the LSTM and get predicted trajectory
        #             predicted_trajectory = self.predictor(paths, scene_goals, n_predict=12, obs_length=9, modes=1, args=self.args)
        #             predicted_trajectory = predicted_trajectory[0][0]

        #             path_msg = self.prediction_to_path(predicted_trajectory, pedestrian_id)

        #             # Create publisher for this pedestrian if not created yet
        #             self.create_publisher(pedestrian_id)

        #             # Publish the predicted path for this pedestrian
        #             self.pred_traj_pubs[pedestrian_id].publish(path_msg)

        #             self.last_iter_completion_time = rospy.Time.now()

        #     # elapsed_time = (self.last_iter_completion_time - loop_init_time).to_sec()  # to get the total time in seconds
        #     # rospy.logerr("vis markers loop time: %.6f seconds", elapsed_time)

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
    rospy.init_node('lstm_node_array')
    lstm = LstmNodeArray()
    rospy.spin()


if __name__ == '__main__':
    main()



# #!/usr/bin/env python3

# import rospy
# import rospkg  # to load ROS package paths dynamically
# import numpy as np
# from collections import deque  # to have fixed-size moving window
# import argparse  # to define some default args for self.predictor() to work

# from visualization_msgs.msg import Marker  # to handle /visualization_markers messages
# from geometry_msgs.msg import Point  # to display predicted trajectory in RViz
# from hrii_person_tracker.msg import PathArray  # format of output trajectories

# from geometry_msgs.msg import PoseStamped
# from nav_msgs.msg import Path  # for predicted trajectories

# import tf

# # LSTM imports
# ## import trajnetplusplustools
# import trajnetbaselines.lstm.trajnet_evaluator as te  # to use LSTM model from Trajnet++
# import trajnetbaselines.lstm.utils
# import torch  # to use tensors


# class LstmNodeArray:

#     def __init__(self):
#         self.parser = argparse.ArgumentParser()
#         self.args, _ = add_arguments(self.parser)

#         self.last_iter_completion_time = rospy.Time.now()
#         self.data_acquisition_period = rospy.Duration(1)  # default value: 1 second

#         self.obs_length = 9
#         # self.observed_window_is_full = False

#         # Store a fixed-size moving window for each pedestrian in a dictionary. Key: pedestrian id, value: fixed-size moving window containing their observed trajectory
#         self.x = {}
#         self.y = {}
        
#         # Dynamically get the model path
#         package_path = rospkg.RosPack().get_path('hrii_person_tracker')
#         self.model = f"{package_path}/scripts/lstm_vanilla_None.pkl"

#         # Load LSTM model using Trajnet++
#         self.predictor = te.load_predictor(self.model)

#         # tf listener to get transform of ($arg robot_id)_front_laser_link wrt ($arg robot_id)_map
#         self.mobile_base = rospy.get_param("~mobile_base")
#         self.global_frame = self.mobile_base + "_map"
#         self.odom_frame = self.mobile_base + "_odom"
#         self.laser_frame = self.mobile_base + "_front_laser_link"
#         self.listener = tf.TransformListener()

#         # Subscriber for visualization markers
#         self.visualization_marker_sub = rospy.Subscriber('/visualization_marker', Marker, self.visualization_markers_callback)

#         # Dynamic publishers for each pedestrian based on their ID
#         self.pred_traj_pubs = {}

#     def create_publisher(self, pedestrian_id):
#         """Create a publisher for a pedestrian's predicted trajectory if it does not exist."""
#         if pedestrian_id not in self.pred_traj_pubs:
#             topic_name = f'/predicted_trajectory{pedestrian_id}'
#             self.pred_traj_pubs[pedestrian_id] = rospy.Publisher(topic_name, Path, queue_size=10)
#             rospy.loginfo(f"Created publisher for {topic_name}")

#     def prediction_to_path(self, predicted_trajectory, pedestrian_id):
#         """Convert predicted trajectory to nav_msgs/Path."""
#         path = Path()
#         path.header.stamp = rospy.Time.now()
#         path.header.frame_id = self.global_frame

#         for point in predicted_trajectory:
#             pose = PoseStamped()
#             pose.header.stamp = rospy.Time.now()
#             pose.header.frame_id = self.global_frame
#             pose.pose.position.x = point[0]
#             pose.pose.position.y = point[1]
#             pose.pose.position.z = 0  # 2D prediction => z = 0
#             pose.pose.orientation.w = 1.0  # no orientation considered
#             path.poses.append(pose)

#         return path

#     def visualization_markers_callback(self, msg):
#         """Callback to handle incoming markers and predict trajectories."""
#         if (rospy.Time.now() - self.last_iter_completion_time < self.data_acquisition_period): #and not(self.observed_window_is_full):
#             return

#         # loop_init_time = rospy.Time.now()

#         if msg.ns == "PEOPLE":
#             pedestrian_id = msg.id

#             # Initialize the fixed-size moving window for the pedestrian if it does not exist
#             if pedestrian_id not in self.x:
#                 self.x[pedestrian_id] = deque(maxlen=self.obs_length)
#                 self.y[pedestrian_id] = deque(maxlen=self.obs_length)
            
#             rospy.logerr("len(self.x[%d]) %d, len(self.y[%d]) %d", pedestrian_id, len(self.x[pedestrian_id]), pedestrian_id, len(self.y[pedestrian_id]))

#             # Get the position of the marker in odom_frame
#             x_odom = msg.pose.position.x
#             y_odom = msg.pose.position.y

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

#             # Check if there are already points in the deque
#             if len(self.x[pedestrian_id]) > 0:
#                 # Get the last recorded point in the global frame
#                 x_last = self.x[pedestrian_id][-1]
#                 y_last = self.y[pedestrian_id][-1]

#                 # Calculate the distance between the msg point (converted in global frame) and the last recorded point
#                 distance = np.sqrt((x_global - x_last) ** 2 + (y_global - y_last) ** 2)

#                 # If the distance is greater than 1 meter, reset the deque
#                 if distance > 2.0:
#                     rospy.logwarn(f"Pedestrian {pedestrian_id} moved more than 1 meter. Resetting trajectory.")
                    
#                     # Fill both x and y deque with the same point to ensure prediction can still be made (it will be just a first prediction, later refined)
#                     self.x[pedestrian_id] = deque([x_global] * self.obs_length, maxlen=self.obs_length)
#                     self.y[pedestrian_id] = deque([y_global] * self.obs_length, maxlen=self.obs_length)
#                     # self.observed_window_is_full = False


#             # Add the transformed points to the pedestrian's trajectory buffer
#             self.x[pedestrian_id].append(x_global)
#             self.y[pedestrian_id].append(y_global)


#             # Predict and publish trajectory if we have enough points for the pedestrian
#             if len(self.x[pedestrian_id]) == self.obs_length:
#                 # self.observed_window_is_full = True

#                 ## Convert from deque to NumPy array
#                 combined = np.array([[self.x[pedestrian_id][i], self.y[pedestrian_id][i]] for i in range(self.obs_length)])
#                 paths = trajnetbaselines.lstm.utils.xy_to_paths(combined)
#                 paths = [paths]
                
#                 scene_goals_np = np.zeros((len(paths), 2))
#                 scene_goals = torch.tensor(scene_goals_np, dtype=torch.float)

#                 ## Feed the LSTM and get predicted trajectory
#                 predicted_trajectory = self.predictor(paths, scene_goals, n_predict=12, obs_length=9, modes=1, args=self.args)
#                 predicted_trajectory = predicted_trajectory[0][0]

#                 path_msg = self.prediction_to_path(predicted_trajectory, pedestrian_id)

#                 # Create publisher for this pedestrian if not created yet
#                 self.create_publisher(pedestrian_id)

#                 # Publish the predicted path for this pedestrian
#                 self.pred_traj_pubs[pedestrian_id].publish(path_msg)

#                 self.last_iter_completion_time = rospy.Time.now()

#         # elapsed_time = (self.last_iter_completion_time - loop_init_time).to_sec()  # to get the total time in seconds
#         # rospy.logerr("vis markers loop time: %.6f seconds", elapsed_time)

# def add_arguments(parser):
#     parser.add_argument('--path', default='DATA_BLOCK/synth_data/test_pred/',
#                         help='directory of data to test')
#     parser.add_argument('--output', nargs='+',
#                         help='relative path to saved model')
#     parser.add_argument('--obs_length', default=9, type=int,
#                         help='observation length')
#     parser.add_argument('--pred_length', default=12, type=int,
#                         help='prediction length')
#     parser.add_argument('--write_only', action='store_true',
#                         help='disable writing new files')
#     parser.add_argument('--disable-collision', action='store_true',
#                         help='disable collision metrics')
#     parser.add_argument('--labels', required=False, nargs='+',
#                         help='labels of models')
#     parser.add_argument('--normalize_scene', action='store_true',
#                         help='augment scenes')
#     parser.add_argument('--modes', default=1, type=int,
#                         help='number of modes to predict')
#     parser.add_argument('--scene_id', default=1888, type=int,
#                         help='scene id')
#     return parser.parse_known_args()


# def main():
#     rospy.init_node('lstm_node_array')
#     lstm = LstmNodeArray()
#     rospy.spin()


# if __name__ == '__main__':
#     main()