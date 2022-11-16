#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  cv::Mat image;
  image =
      cv::imread("/home/xxt/bisenet_ws/src/bisenet_ros/test/img/example.png");
  // cv::cvtColor(image, image, CV_BGR2GRAY);
  // cv::imshow("GRAY", image);

  cv::imshow("BGR", image);
  cv::cvtColor(image, image, CV_BGR2RGB);
  cv::imshow("RGB", image);
  cv::waitKey();
  return 0;
}
