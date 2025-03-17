#!/usr/bin/env python3

import rospy
from std_srvs.srv import Empty

rospy.wait_for_service('/gazebo/unpause_physics')
unpause_physics_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
unpause_physics_service()