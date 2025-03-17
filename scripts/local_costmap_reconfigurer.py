#!/usr/bin/env python

import rospy
from dynamic_reconfigure.client import Client
from nav_msgs.msg import Path
from hrii_person_tracker.msg import PathArray

class CostmapLayerManager:
    def __init__(self):
        rospy.init_node('costmap_layer_manager')

        # Initialize dynamic reconfigure clients
        self.obstacle_client = Client('/rbkairos/move_base/local_costmap/obstacle_laser_layer', timeout=30)
        # self.lstm_client = Client('/rbkairos/move_base/local_costmap/lstm_layer', timeout=30)
        self.lstm_client = Client('/rbkairos/move_base/local_costmap/lstm_layer_array', timeout=30)
        
        # Subscriber to /predicted_trajectory topic
        # self.trajectory_sub = rospy.Subscriber('/predicted_trajectory', Path, self.trajectory_callback)
        self.trajectory_sub = rospy.Subscriber('/predicted_trajectory', PathArray, self.trajectory_callback)

        # Initial states
        self.trajectory_available = False

        # Setup a timer to check the availability of the trajectory periodically
        self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)

    def trajectory_callback(self, msg):
        # Callback function for when new messages arrive on /predicted_trajectory
        self.trajectory_available = True
        self.update_costmap_layers()

    def timer_callback(self, event):
        # Timer callback to check for the availability of trajectory messages
        if not self.trajectory_available:
            self.update_costmap_layers()

        # Reset the flag after checking
        self.trajectory_available = False

    def update_costmap_layers(self):
        # Update the costmap layers based on the availability of trajectory messages
        if self.trajectory_available:
            rospy.loginfo("LSTM Layer ON, Obstacle Layer OFF")
            self.lstm_client.update_configuration({'enabled': True})
            self.obstacle_client.update_configuration({'enabled': False})
        else:
            rospy.loginfo("LSTM Layer OFF Obstacle Layer ON")
            self.lstm_client.update_configuration({'enabled': False})
            self.obstacle_client.update_configuration({'enabled': True})

if __name__ == '__main__':
    try:
        layer_manager = CostmapLayerManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
