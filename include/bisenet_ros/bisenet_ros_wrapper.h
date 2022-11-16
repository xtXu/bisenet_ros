#include <ros/ros.h>
#include <string>

#include <torch/script.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Eigen>

#include <image_transport/image_transport.h>

class BisenetRosWrapper {
public:
private:
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_label_pub_;
  image_transport::Publisher image_rgb_pub_;

  std::string module_path_;
  torch::Tensor mean_;
  torch::Tensor std_;
  unsigned char color_[256][3];

  torch::jit::script::Module module_;

public:
  BisenetRosWrapper(ros::NodeHandle &nh, ros::NodeHandle &nh_private);
  ~BisenetRosWrapper();

  void init(ros::NodeHandle &nh, ros::NodeHandle &nh_private);
  cv::Mat inference(cv::Mat img);

  cv::Mat &loadImage(const std::string img_path, cv::Mat &img);
  void showImage(cv::Mat &img, std::string title);
  cv::Mat &generateSemRGB(const cv::Mat &semantic_img, cv::Mat &semantic_rgb);

private:
  void loadTorchModule();
  void generateLabelColor();

  cv::Mat &preprocessImage(cv::Mat &img);
  cv::Mat &inferenceSemSeg(const torch::Tensor &input, cv::Mat &semantic_img,
                           cv::Size size);

  torch::Tensor &creatTensorFromImage(const cv::Mat &img,
                                      torch::Tensor &tensor);

	void imageInferCallback(const sensor_msgs::ImageConstPtr& msg);
};
