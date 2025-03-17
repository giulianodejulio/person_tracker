#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped

class CircularBuffer:
    def __init__(self):
        self.previous_value = None  # This will store the previous value (u3)
    
    def switch_and_memory(self, u1, u2):
        """
        Simulates the circular buffer behavior with a Switch block and Memory block.
        
        Parameters:
        - u1: New value to be passed when u2 != 0
        - u2: Condition to check (if u2 != 0, output u1, else output previous value)
        
        Returns:
        - y: Output of the circular buffer
        """
        if u2 != 0:
            y = u1
        else:
            y = self.previous_value if self.previous_value is not None else 0  # If u2 is zero, set y to previous value
        
        self.previous_value = y  # Update the previous value (u3)
        
        return y


# Global variables to store the previous position and timestamp
previous_position = Point()
previous_time = rospy.Time()
# velocity_publisher = None  # To store the publisher object
circular_buffer = CircularBuffer()


# Function to calculate the dirty derivative (velocity) based on the change in position over time
def compute_velocity(current_position, current_time):
    global previous_position, previous_time
    
    # Calculate time difference (delta_t)
    delta_t = (current_time - previous_time).to_sec()  # Convert to seconds
    
    # Avoid division by zero (when delta_t is too small)
    if delta_t > 0:
        # Calculate the velocity in x and y directions (dirty derivative)
        velocity_x = (current_position.x - previous_position.x) / delta_t
        velocity_y = (current_position.y - previous_position.y) / delta_t
        
        # Filter the velocity using a circular buffer
        velocity_x = circular_buffer.switch_and_memory(velocity_x, velocity_x)
        velocity_y = circular_buffer.switch_and_memory(velocity_y, velocity_y)

        # Update previous position and time
        previous_position = current_position
        previous_time = current_time
        
        # Log the velocity
        rospy.loginfo("Velocity - X: %.2f m/s, Y: %.2f m/s", velocity_x, velocity_y)
        
        # Publish the velocity as a PointStamped message
        velocity_msg = PointStamped()
        velocity_msg.header.stamp = current_time
        velocity_msg.header.frame_id = "moca_red_odom"
        velocity_msg.point.x = velocity_x
        velocity_msg.point.y = velocity_y
        velocity_msg.point.z = 0.0  # No velocity in the z-direction
        
        velocity_publisher.publish(velocity_msg)
    # else:
    #     rospy.logwarn("Time difference is too small: %.6f seconds", delta_t)

# Callback function for the /visualization_marker topic
def visualization_marker_callback(marker):
    if marker.ns == "People_tracked" and marker.points:
        x_odom = 0.5*(marker.points[0].x + marker.points[1].x)
        y_odom = 0.5*(marker.points[0].y + marker.points[1].y)

    # with self.data_lock:
    #     # Get the position of the center of the legs in odom frame
    #     x_odom = 0.5*(msg.points[0].x + msg.points[1].x)
    #     y_odom = 0.5*(msg.points[0].y + msg.points[1].y)

    #     pedestrian_id = int(msg.text)

    #     # Get the transform from odom frame to global frame
    #     try:
    #         now = rospy.Time.now()
    #         self.listener.waitForTransform(self.global_frame, self.odom_frame, now, rospy.Duration(1.0))
    #         (trans, rot) = self.listener.lookupTransform(self.global_frame, self.odom_frame, now)
    #     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #         rospy.logerr("TF Exception occurred")
    #         return

    #     # Create a transformation matrix from the obtained translation and rotation
    #     transform_matrix = tf.transformations.quaternion_matrix(rot)
    #     transform_matrix[0:3, 3] = trans

    #     # Convert the odom coordinates to homogeneous coordinates
    #     point_local = np.array([x_odom, y_odom, 0, 1])

    #     # Transform the point from the odom frame to the global frame
    #     point_global = np.dot(transform_matrix, point_local)
    #     x_global, y_global = point_global[0], point_global[1]

        # Extract position and timestamp from the Marker message
        current_position = Point()
        current_position.x = x_odom
        current_position.y = y_odom
        current_time = rospy.Time.now()

        # Compute and log the velocity (dirty derivative)
        compute_velocity(current_position, current_time)
    # else:
    #     rospy.logerr("visualization_marker message not received")

def listener():
    global velocity_publisher

    # Initialize the ROS node
    rospy.init_node('dirty_derivative_node', anonymous=True)
    
    # Initialize the publisher for /dirty_derivative
    velocity_publisher = rospy.Publisher('/dirty_derivative', PointStamped, queue_size=10)
    
    # Subscribe to the /visualization_marker topic
    rospy.Subscriber('/visualization_marker', Marker, visualization_marker_callback)
    
    # Set a default loop rate (doesn't affect the actual dirty-derivative-computation loop)
    rate = rospy.Rate(1)

    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    try:
        # Run the listener function
        listener()
    except rospy.ROSInterruptException:
        pass
