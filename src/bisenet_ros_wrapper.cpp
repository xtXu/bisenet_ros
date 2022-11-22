#include <bisenet_ros/bisenet_ros_wrapper.h>
#include <cv_bridge/cv_bridge.h>

BisenetRosWrapper::BisenetRosWrapper(ros::NodeHandle &nh,
                                     ros::NodeHandle &nh_private)
    : it_(nh) {}

BisenetRosWrapper::~BisenetRosWrapper() {}

void BisenetRosWrapper::init(ros::NodeHandle &nh, ros::NodeHandle &nh_private) {
  nh_private.param("torch/module_path", module_path_, std::string(""));

  double mean_rgb[3], std_rgb[3];
  nh_private.param("cv/mean_r", mean_rgb[0], -1.0);
  nh_private.param("cv/mean_g", mean_rgb[1], -1.0);
  nh_private.param("cv/mean_b", mean_rgb[2], -1.0);
  nh_private.param("cv/std_r", std_rgb[0], -1.0);
  nh_private.param("cv/std_g", std_rgb[1], -1.0);
  nh_private.param("cv/std_b", std_rgb[2], -1.0);

  nh_private.param("use_const_mean_std", use_const_mean_std_, true);

  mean_ = torch::from_blob(mean_rgb, {3, 1, 1}, torch::kFloat64)
              .clone()
              .toType(torch::kFloat);
  std_ = torch::from_blob(std_rgb, {3, 1, 1}, torch::kFloat64)
             .clone()
             .toType(torch::kFloat);

  image_sub_ = it_.subscribe("/input_image", 1,
                             &BisenetRosWrapper::imageInferCallback, this);
  image_label_pub_ = it_.advertise("/output_image_label", 1);
  image_rgb_pub_ = it_.advertise("/output_image_rgb", 1);

  loadTorchModule();
  generateLabelColor();
}

void BisenetRosWrapper::loadTorchModule() {
  try {
    module_ = torch::jit::load(module_path_);
    module_.eval(); // eval mode
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    exit(-1);
  }
}

cv::Mat &BisenetRosWrapper::loadImage(const std::string img_path,
                                      cv::Mat &img) {
  img = cv::imread(img_path);
  return img;
}

cv::Mat BisenetRosWrapper::inference(cv::Mat img) {

  torch::Tensor tensor_img;
  cv::Mat semantic_img;
  int org_size[2] = {img.size[0], img.size[1]};

  img = preprocessImage(img);
  tensor_img = creatTensorFromImage(img, tensor_img);
  semantic_img = inferenceSemSeg(tensor_img, semantic_img,
                                 cv::Size(org_size[1], org_size[0]));

  return semantic_img;
}

cv::Mat &BisenetRosWrapper::preprocessImage(cv::Mat &img) {

  // make the new size proper for torch
  int new_size[2];
  new_size[0] = std::ceil(img.size[0] / 32.0) * 32;
  new_size[1] = std::ceil(img.size[1] / 32.0) * 32;
  cv::resize(img, img, cv::Size(new_size[1], new_size[0]));

  // cv: BGR  torch: RGB
  cv::cvtColor(img, img, CV_BGR2RGB);

  return img;
}

torch::Tensor &BisenetRosWrapper::creatTensorFromImage(const cv::Mat &img,
                                                       torch::Tensor &tensor) {
  // Creat tensor from the cv::Mat
  // cv: H*W*C -> torch: C*H*W
  // use .clone(), to copy the tensor data
  // otherwise, change the vector data will change the tensor
  tensor =
      torch::from_blob(img.data, {img.size[0], img.size[1], 3}, torch::kByte)
          .clone();
  tensor = tensor.permute({2, 0, 1});
  tensor = tensor.toType(torch::kFloat);

  // normalize to [0,1]
  tensor = tensor.div_(255);

  // z-score normalize
  // torch::Tensor mean_tensor = torch::from_blob(mean_.data(), {3,1,1},
  // torch::kFloat).clone(); torch::Tensor std_tensor =
  // torch::from_blob(std_.data(), {3,1,1}, torch::kFloat).clone();
	if (use_const_mean_std_) {
		tensor = tensor.sub_(mean_).div_(std_);
	} else {
		torch::Tensor mean = torch::mean(tensor, {2,1}, true);
		torch::Tensor std = torch::std(tensor, {2,1}, true, true);
		tensor = tensor.sub_(mean).div_(std);
	}

  // add a dimension for batch
  // because torch need B*C*H*W
  tensor = tensor.unsqueeze(0);

  return tensor;
}

cv::Mat &BisenetRosWrapper::inferenceSemSeg(const torch::Tensor &input,
                                            cv::Mat &semantic_img,
                                            cv::Size size) {
  // creat a torch module input
  std::vector<torch::jit::IValue> tensor_input;
  tensor_input.push_back(input);

  // inference
  torch::Tensor output;
  output = module_.forward(tensor_input).toTensor();
  // std::cout << "inference ok \n";

  output = output.squeeze(); // delete the batch dimension
  output = output.detach().to(torch::kU8);

  semantic_img = cv::Mat(cv::Size(output.size(1), output.size(0)), CV_8U,
                         output.data_ptr())
                     .clone();
  cv::resize(semantic_img, semantic_img, size);

  return semantic_img;
}

cv::Mat &BisenetRosWrapper::generateSemRGB(const cv::Mat &semantic_img,
                                           cv::Mat &semantic_rgb) {
  int height = semantic_img.size[0];
  int width = semantic_img.size[1];
  uchar semantnc_img_data[height][width][3];
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      uchar label = semantic_img.at<uchar>(h, w);
      semantnc_img_data[h][w][0] = color_[label][0];
      semantnc_img_data[h][w][1] = color_[label][1];
      semantnc_img_data[h][w][2] = color_[label][2];
    }
  }
  semantic_rgb =
      cv::Mat(cv::Size(width, height), CV_8UC3, semantnc_img_data).clone();

	cv::cvtColor(semantic_rgb, semantic_rgb, CV_RGB2BGR);

  return semantic_rgb;
}

void BisenetRosWrapper::imageInferCallback(
    const sensor_msgs::ImageConstPtr &msg) {
	
	// ROS_INFO("callback");
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

	cv::Mat semantic_img, semantic_rgb;
	semantic_img = inference(cv_ptr->image);
	semantic_rgb = generateSemRGB(semantic_img, semantic_rgb);

	sensor_msgs::ImagePtr label_msg, rgb_msg;
	label_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", semantic_img).toImageMsg();
	rgb_msg = cv_bridge::CvImage(std_msgs::Header(),"bgr8", semantic_rgb).toImageMsg();

	image_label_pub_.publish(label_msg);
	image_rgb_pub_.publish(rgb_msg);
	// ROS_INFO("callback_finish");
}

void BisenetRosWrapper::generateLabelColor() {
  // the same seed generate the same color table
  srand(123);
  for (auto &a : color_) {
    for (auto &b : a) {
      b = rand() % 256;
    }
  }

	color_[255][0] = 0;
	color_[255][1] = 0;
	color_[255][2] = 0;

	color_[10][0] = 255;
	color_[10][1] = 255;
	color_[10][2] = 255;
}

void BisenetRosWrapper::showImage(cv::Mat &img, std::string title) {
  cv::namedWindow("image", true);
  cv::imshow("image", img);
  cv::waitKey();
  cv::destroyAllWindows();
}
