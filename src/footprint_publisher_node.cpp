#include <ros/ros.h>
#include <geometry_msgs/PolygonStamped.h>

class FootprintRepublisher {
public:
    FootprintRepublisher(ros::NodeHandle& nh) {
        // Subscribe to the footprint topic
        footprint_sub_ = nh.subscribe("/rbkairos/move_base/global_costmap/footprint", 10, &FootprintRepublisher::footprintCallback, this);
        // Publish on the new topic
        footprint_pub_ = nh.advertise<geometry_msgs::PolygonStamped>("/footprint", 10);
    }

private:
    void footprintCallback(const geometry_msgs::PolygonStamped::ConstPtr& msg) {
        // Publish the received footprint
        footprint_pub_.publish(msg);
        // ROS_INFO("Published footprint data.");
    }

    ros::Subscriber footprint_sub_;
    ros::Publisher footprint_pub_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "footprint_publisher_node");
    ros::NodeHandle nh;

    FootprintRepublisher republisher(nh);

    ros::spin();
    return 0;
}

