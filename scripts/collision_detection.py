#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from leg_tracker.msg import People
from hrii_person_tracker.msg import PathArray

import numpy as np

class CollisionDetection:
    def __init__(self):
        self.people_sub = rospy.Subscriber('/people', People, self.people_callback)
        self.lstm_node_array_sub = rospy.Subscriber('/predicted_trajectory', PathArray, self.lstm_node_array_callback)
        self.lstm_output_kalman_timestamps_pub_ = rospy.Publisher('/lstm_output_kalman_timestamps', Float64MultiArray, queue_size=10)

        self.people_current_positions = {}
        self.people_velocities = {}
        self.predicted_paths = {}


    def people_callback(self, msg):
        for person in msg.people:
            person_id = person.name
            self.people_current_positions[person_id] = [person.position.x, person.position.y, person.position.z]
            self.people_velocities[person_id]        = [person.velocity.x, person.velocity.y, person.velocity.z]


    def lstm_node_array_callback(self, msg):

        # remove outdated pedestrians data
        pedestrians_to_remove = []
        for pedestrian_id in self.predicted_paths:
            if(rospy.Time.now () - self.pedestrian_data[pedestrian_id]['last_update_time'] >= self.deprecated_data_time):
                pedestrians_to_remove.append(pedestrian_id)
        for pedestrian_id in pedestrians_to_remove:
            rospy.loginfo("Deleting ped %d (deprecated)", pedestrian_id)
            del self.pedestrian_data[pedestrian_id]
        for path in msg.path:
            self.predicted_paths[msg.pedestrian_id] = path

        print(self.people_velocities)
        # print(self.people_velocities[person_id])
        
        # self.compute_predicted_trajectory_timestamps()


    def compute_predicted_trajectory_timestamps(self):
        for person_id, path in self.predicted_paths.items():
            # Extract current position and velocity of the person
            current_position = np.array(self.people_current_positions.get(person_id, [0.0, 0.0, 0.0]))
            # velocity         = np.array(self.people_velocities.get(person_id, [0.0, 0.0, 0.0]))
            velocity         = np.array(self.people_velocities.get(person_id))
            print("velocity ", velocity)

            # # Calculate the velocity magnitude
            # # velocity_magnitude = np.linalg.norm(velocity[:2])  # Use x, y components for velocity computation
            # velocity_magnitude = np.linalg.norm(velocity)  # Use x, y components for velocity computation

            # if velocity_magnitude == 0:
            #     rospy.logwarn(f"Person {person_id} has zero velocity; skipping timestamp computation.")
            #     continue

            # # Initialize matrix to store positions and timestamps
            # trajectory_data = []
            # previous_position = current_position[:2]  # Only x, y components for 2D
            # total_time = 0.0  # Cumulative time for timestamps

            # # Iterate through the path's poses
            # for pose in path.poses:
            #     # Get the 2D position from the pose
            #     current_pose_position = np.array([pose.pose.position.x, pose.pose.position.y])

            #     # Compute the Euclidean distance between current and previous position
            #     distance = np.linalg.norm(current_pose_position - previous_position)

            #     # Compute timestamp increment based on velocity
            #     time_increment = distance / velocity_magnitude
            #     total_time += time_increment

            #     # Append the 2D position and timestamp to the matrix
            #     trajectory_data.append([current_pose_position[0], current_pose_position[1], total_time])

            #     # Update the previous position
            #     previous_position = current_pose_position

            # # Publish the trajectory data
            # self.publish_trajectory_data(person_id, trajectory_data)
    
    
    def publish_trajectory_data(self, person_id, trajectory_data):
        # Create a Float64MultiArray message
        msg = Float64MultiArray()

        # Set the layout (dimensions of the matrix)
        msg.layout.dim = [
            MultiArrayDimension(label='rows', size=len(trajectory_data), stride=3 * len(trajectory_data)),
            MultiArrayDimension(label='columns', size=3, stride=3)
        ]
        msg.layout.data_offset = 0

        # Flatten the trajectory data matrix into a single list
        flat_data = [item for sublist in trajectory_data for item in sublist]
        msg.data = flat_data

        # Log and publish the message
        rospy.loginfo(f"Publishing trajectory data for person {person_id}: {msg}")
        self.lstm_output_kalman_timestamps_pub_.publish(msg)


def main():
    rospy.init_node('collision_detection')
    collision_detection = CollisionDetection()
    rospy.spin()

if __name__ == '__main__':
    main()