#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "videototopic");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  image_transport::ImageTransport it(nh); // 发布图片需要用到image_transport
  image_transport::Publisher pub = it.advertise("/usb_cam/image_raw", 1);

  ros::Rate loop_rate(30);

  std::string path;
  nh_private.param("video_path", path, std::string(""));

  std::cout << path << std::endl;
  cv::VideoCapture cap(path); // open video from the path
  if (!cap.isOpened()) {
    std::cerr << "open video failed!" << std::endl;
    return -1;
  } else {
    std::cout << "open video success!" << std::endl;
  }

  cv::Mat frame; // this is an image
  bool isSuccess = true;
  while (nh.ok()) {
    isSuccess = cap.read(frame);
    if (!isSuccess) { // if the video ends, then break
      std::cout << "video ends" << std::endl;
      break;
    }
    // 将opencv的图片转换成ros的sensor_msgs，然后才能发布。
    sensor_msgs::ImagePtr msg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
    pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}
