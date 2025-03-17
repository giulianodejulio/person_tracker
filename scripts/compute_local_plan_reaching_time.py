#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
import tf2_ros
import tf2_geometry_msgs
from math import sqrt

class LocalPlanAnalyzer:
    def __init__(self):
        rospy.init_node('local_plan_analyzer', anonymous=True)
        # Subscriber to the local plan
        self.local_plan_sub = rospy.Subscriber('/moca_red/move_base/TebLocalPlannerROS/local_plan', Path, self.local_plan_callback)
        # Subscriber to the cmd_vel topic
        self.cmd_vel_sub = rospy.Subscriber('/moca_red/move_base/cmd_vel', Twist, self.cmd_vel_callback)
        # Publish only the first local plan
        self.first_local_plan_pub = rospy.Publisher('/first_local_plan', Path, queue_size=10)
        # TF2 Buffer and Listener
        self.global_frame = "moca_red_map"
        self.base_footprint_frame = "moca_red_base_footprint"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # To track whether the first message has been processed
        self.first_message_processed = False
        # Current velocity from /cmd_vel
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.local_plan_poses = None


    def cmd_vel_callback(self, msg):
        """
        Callback to update current robot velocity from /cmd_vel topic.
        """
        self.current_linear_velocity = msg.linear.x
        self.current_angular_velocity = msg.angular.z
        self.first_local_plan_pub.publish(self.first_local_plan)
        if self.local_plan_poses:
            self.calculate_times_to_poses()


    def local_plan_callback(self, msg):
        """
        Callback for the local plan topic. Processes the first message and calculates timestamps.
        """
        if self.first_message_processed:
            return  # Ignore subsequent messages
        self.first_message_processed = True
        rospy.loginfo(f"Received the first local plan message with {len(msg.poses)} positions")
        # Get the initial time of the plan
        self.initial_time = rospy.Time.now()
        rospy.loginfo("Current time:")
        rospy.loginfo(f"{self.initial_time.to_sec()} s")

        self.local_plan_poses = msg.poses
        self.idx = 1
        self.next_pose_position = [self.local_plan_poses[self.idx].pose.position.x, self.local_plan_poses[self.idx].pose.position.y]
        
        self.first_local_plan = Path()
        self.first_local_plan.header = msg.header
        self.first_local_plan.poses = self.local_plan_poses
        # rospy.loginfo("Published the first local plan on /first_local_plan")


        # rospy.loginfo("self.local_plan_poses[0]: ")
        # rospy.loginfo(f"Position (x={self.local_plan_poses[0].pose.position.x}, y={self.local_plan_poses[0].pose.position.x})")

        # rospy.loginfo("self.local_plan_poses: ")
        # for i, pose_stamped in enumerate(self.local_plan_poses):
        #     position = pose_stamped.pose.position
            # orientation = pose_stamped.pose.orientation
            # rospy.loginfo(f"Pose {i}: Position (x={position.x}, y={position.y}, z={position.z}), "
            #               f"Orientation (x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w})")


    def calculate_times_to_poses(self):
        """
        Calculate the time at which the base_link overlaps with each pose.
        """
        try:
            # Lookup the transform
            transform: TransformStamped = self.tf_buffer.lookup_transform(self.global_frame, self.base_footprint_frame, rospy.Time(0))
            base_footprint_position_in_map = transform.transform.translation
            # rot = transform.transform.rotation
            # rospy.loginfo(f"Rotation: x={rot.x}, y={rot.y}, z={rot.z}, w={rot.w}")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Transform lookup failed: {e}")
        
        dx = base_footprint_position_in_map.x - self.next_pose_position[0]
        dy = base_footprint_position_in_map.y - self.next_pose_position[1]

        if (sqrt(dx**2 + dy**2) <= 0.01):
            now = rospy.Time.now()
            rospy.loginfo(f"{now.to_sec()} s")
            self.next_pose_position = [self.local_plan_poses[self.idx+1].pose.position.x, self.local_plan_poses[self.idx+1].pose.position.y]
            self.idx += 1




if __name__ == '__main__':
    try:
        LocalPlanAnalyzer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
