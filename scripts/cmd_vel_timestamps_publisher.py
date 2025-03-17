#!/usr/bin/env python3

import rospy
from teb_local_planner_2.msg import FeedbackMsg
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class TebFeedbackVisualizer:
    def __init__(self):
        # Node initialization
        rospy.init_node('cmd_vel_timestamps_publisher', anonymous=True)

        # Subscriber to the teb feedback topic
        self.teb_feedback_sub = rospy.Subscriber(
            '/moca_red/move_base/TebLocalPlannerROS/teb_feedback',
            FeedbackMsg,
            self.teb_feedback_callback
        )

        # Publisher for the visualization markers
        self.marker_pub = rospy.Publisher(
            '/teb_feedback_markers',
            MarkerArray,
            queue_size=10
        )

        rospy.loginfo("Teb Feedback Visualizer Node Started")

    def teb_feedback_callback(self, msg):
        # Create a MarkerArray to publish
        marker_array = MarkerArray()

        # Iterate through the trajectories
        for traj_idx, trajectory in enumerate(msg.trajectories):
            # Iterate through the trajectory points
            for point_idx, trajectory_point in enumerate(trajectory.trajectory):
                # Create a text marker
                marker = Marker()
                marker.header.frame_id = trajectory.header.frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = f"trajectory_{traj_idx}"
                marker.id = point_idx + traj_idx * 100  # Unique ID for each marker
                marker.type = Marker.TEXT_VIEW_FACING
                marker.action = Marker.ADD

                # Position the marker at the trajectory point
                marker.pose.position = trajectory_point.pose.position
                marker.pose.position.z += 0.2  # Slightly elevate text for better visibility

                # Orientation (not relevant for text)
                marker.pose.orientation.w = 1.0

                # Set the text as time_from_start
                marker.text = f"{trajectory_point.time_from_start.to_sec():.2f} s"

                # Marker scale and color
                marker.scale.z = 0.5  # Text height
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                # Add the marker to the array
                marker_array.markers.append(marker)

        # Publish the MarkerArray
        self.marker_pub.publish(marker_array)

if __name__ == "__main__":
    try:
        visualizer = TebFeedbackVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass




# # The following version subtracts the current time from the timestamps, hence it must be employed when the timestamps are absolute

# # !/usr/bin/env python3

# import rospy
# from teb_local_planner_2.msg import FeedbackMsg
# from visualization_msgs.msg import Marker, MarkerArray
# from geometry_msgs.msg import Point

# class TebFeedbackVisualizer:
#     def __init__(self):
#         # Node initialization
#         rospy.init_node('teb_feedback_visualizer', anonymous=True)

#         # Subscriber to the teb feedback topic
#         self.teb_feedback_sub = rospy.Subscriber(
#             '/moca_red/move_base/TebLocalPlannerROS/teb_feedback',
#             FeedbackMsg,
#             self.teb_feedback_callback
#         )

#         # Publisher for the visualization markers
#         self.marker_pub = rospy.Publisher(
#             '/teb_feedback_markers',
#             MarkerArray,
#             queue_size=10
#         )

#         rospy.loginfo("Teb Feedback Visualizer Node Started")

#     def teb_feedback_callback(self, msg):
#         # Create a MarkerArray to publish
#         marker_array = MarkerArray()

#         # Current time
#         current_time = rospy.Time.now()

#         # Iterate through the trajectories
#         for traj_idx, trajectory in enumerate(msg.trajectories):
#             # Iterate through the trajectory points
#             for point_idx, trajectory_point in enumerate(trajectory.trajectory):
#                 # Subtract the current time from time_from_start
#                 time_difference = trajectory_point.time_from_start.to_sec() - current_time.to_sec()

#                 # Create a text marker
#                 marker = Marker()
#                 marker.header.frame_id = trajectory.header.frame_id
#                 marker.header.stamp = rospy.Time.now()
#                 marker.ns = f"trajectory_{traj_idx}"
#                 marker.id = point_idx + traj_idx * 100  # Unique ID for each marker
#                 marker.type = Marker.TEXT_VIEW_FACING
#                 marker.action = Marker.ADD

#                 # Position the marker at the trajectory point
#                 marker.pose.position = trajectory_point.pose.position
#                 marker.pose.position.z += 0.2  # Slightly elevate text for better visibility

#                 # Orientation (not relevant for text)
#                 marker.pose.orientation.w = 1.0

#                 # Set the text as the time difference
#                 marker.text = f"{time_difference:.2f} s"

#                 # Marker scale and color
#                 marker.scale.z = 0.25  # Text height
#                 marker.color.r = 1.0
#                 marker.color.g = 1.0
#                 marker.color.b = 0.0
#                 marker.color.a = 1.0

#                 # Add the marker to the array
#                 marker_array.markers.append(marker)

#         # Publish the MarkerArray
#         self.marker_pub.publish(marker_array)

# if __name__ == "__main__":
#     try:
#         visualizer = TebFeedbackVisualizer()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass
