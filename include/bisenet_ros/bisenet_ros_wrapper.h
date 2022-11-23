#include <ros/ros.h>
#include <string>

#include <torch/script.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Eigen>

#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <sensor_msgs/Image.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class BisenetRosWrapper {
public:
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image>
      ApproximateTimePolicy;

private:
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_label_pub_;
  image_transport::Publisher image_rgb_pub_;

  std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> rgb_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
  std::shared_ptr<message_filters::Synchronizer<ApproximateTimePolicy>> sync_;

	ros::Publisher pcl_pub_;

  std::string module_path_;
  torch::Tensor mean_;
  torch::Tensor std_;
  unsigned char color_[256][4];

  torch::jit::script::Module module_;

  bool use_const_mean_std_;
  bool use_color_map_;
	bool generate_semantic_pcl_;

  std::string color_file_;
  // std::vector<std::vector<uint8_t>> color_map_;

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

  void imageInferCallback(const sensor_msgs::ImageConstPtr &msg);

  void imgDepthRgbCallback(const sensor_msgs::ImageConstPtr &depth,
                           const sensor_msgs::ImageConstPtr &rgb);

  void depthRgba2Pcl(const cv::Mat &depth, const cv::Mat &rgba,
                     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr semantic_pcl);

  void loadColorMap();
};
