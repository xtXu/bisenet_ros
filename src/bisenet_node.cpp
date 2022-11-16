#include <ros/ros.h>
#include <bisenet_ros/bisenet_ros_wrapper.h>

int main (int argc, char *argv[])
{
  ros::init(argc, argv, "bisenet_node");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  BisenetRosWrapper bisenet_ros_wrapper(nh, nh_private);
  bisenet_ros_wrapper.init(nh, nh_private);

	ros::Duration(1.0).sleep();
	ros::spin();

	return 0;
}
