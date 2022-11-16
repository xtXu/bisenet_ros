#include <iostream>
#include <ros/ros.h>

#include <bisenet_ros/bisenet_ros_wrapper.h>

int main(int argc, char *argv[]) {

  ros::init(argc, argv, "test_wrapper");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  BisenetRosWrapper bisenet_ros_wrapper(nh, nh_private);
  bisenet_ros_wrapper.init(nh, nh_private);

  cv::Mat img, semantic_img, semantic_rgb;
  std::string img_path =
      "/home/xxt/bisenet_ws/src/bisenet_ros/test/img/example.png";
  img = bisenet_ros_wrapper.loadImage(img_path, img);
  semantic_img = bisenet_ros_wrapper.inference(img);
  semantic_rgb = bisenet_ros_wrapper.generateSemRGB(semantic_img, semantic_rgb);

  bisenet_ros_wrapper.showImage(semantic_rgb, "semantic");

  return 0;
}
