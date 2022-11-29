#include <bisenet_ros/bisenet_ros_wrapper.h>
#include <bisenet_ros/csv_iterator.h>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <glog/logging.h>

#include <pcl_conversions/pcl_conversions.h>

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

  nh_private.param("cv/use_const_mean_std", use_const_mean_std_, true);
  nh_private.param("cv/use_color_map", use_color_map_, false);

  nh_private.param("cv/color_file", color_file_, std::string(""));

  nh_private.param("pcl/generate_semantic_pcl", generate_semantic_pcl_, false);
  nh_private.param("pcl/maximum_distance", maximum_distance_, -1.0);
  nh_private.param("pcl/flatten_distance", flatten_distance_, -1.0);

  mean_ = torch::from_blob(mean_rgb, {3, 1, 1}, torch::kFloat64)
              .clone()
              .toType(torch::kFloat);
  std_ = torch::from_blob(std_rgb, {3, 1, 1}, torch::kFloat64)
             .clone()
             .toType(torch::kFloat);

  image_label_pub_ = it_.advertise("/output_image_label", 1);
  image_rgb_pub_ = it_.advertise("/output_image_rgb", 1);

  if (generate_semantic_pcl_) {

    depth_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(
        nh, "/input_depth_image", 5));
    rgb_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(
        nh, "/input_rgb_image", 5));
    sync_.reset(new message_filters::Synchronizer<ApproximateTimePolicy>(
        ApproximateTimePolicy(5), *depth_sub_, *rgb_sub_));
    sync_->registerCallback(
        boost::bind(&BisenetRosWrapper::imgDepthRgbCallback, this, _1, _2));

    pcl_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/semantic_pcl", 5);
  } else {
    image_sub_ = it_.subscribe("/input_rgb_image", 1,
                               &BisenetRosWrapper::imageInferCallback, this);
  }

  int i = 0;
  while (!nh.hasParam("/unreal/unreal_ros_client/camera_params/width")) {
    usleep(100000);
    if (++i > 50) {
      ROS_ERROR("Can't load camera_params from "
                "/unreal/unreal_ros_client/camera_params");
      break;
    }
  }
  camera_params_ = {
      nh.param("/unreal/unreal_ros_client/camera_params/width", 640.0),
      nh.param("/unreal/unreal_ros_client/camera_params/height", 480.0),
      nh.param("/unreal/unreal_ros_client/camera_params/focal_length", 320.0)};

  loadTorchModule();

  if (use_color_map_) {
    loadColorMap();
  } else {
    generateLabelColor();
  }
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
    torch::Tensor mean = torch::mean(tensor, {2, 1}, true);
    torch::Tensor std = torch::std(tensor, {2, 1}, true, true);
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
  uchar semantnc_img_data[height][width][4];
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      uchar label = semantic_img.at<uchar>(h, w);
      semantnc_img_data[h][w][0] = color_[label][0];
      semantnc_img_data[h][w][1] = color_[label][1];
      semantnc_img_data[h][w][2] = color_[label][2];
      if (!use_color_map_) {
        semantnc_img_data[h][w][3] = 255;
      } else {
        semantnc_img_data[h][w][3] = color_[label][3];
      }
    }
  }
  semantic_rgb =
      cv::Mat(cv::Size(width, height), CV_8UC4, semantnc_img_data).clone();

  cv::cvtColor(semantic_rgb, semantic_rgb, CV_RGBA2BGRA);

  return semantic_rgb;
}

void BisenetRosWrapper::imgDepthRgbCallback(
    const sensor_msgs::ImageConstPtr &depth,
    const sensor_msgs::ImageConstPtr &rgb) {

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(rgb, "bgr8");
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat semantic_img, semantic_rgb;
  semantic_img = inference(cv_ptr->image);
  semantic_rgb = generateSemRGB(semantic_img, semantic_rgb);

  sensor_msgs::ImagePtr label_msg, rgb_msg;
  label_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", semantic_img)
                  .toImageMsg();
  rgb_msg = cv_bridge::CvImage(std_msgs::Header(), "bgra8", semantic_rgb)
                .toImageMsg();

  image_label_pub_.publish(label_msg);
  image_rgb_pub_.publish(rgb_msg);

  cv_bridge::CvImagePtr depth_ptr;
  try {
    depth_ptr = cv_bridge::toCvCopy(depth, "32FC1");
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

	if (flatten_distance_ > 0.0) {
		double r = depth_ptr->image.rows;
		double c = depth_ptr->image.cols;
		float *pix = depth_ptr->image.ptr<float>(0);
		for (int i = 0; i < r * c; i++) {
			*pix = std::min(*pix, static_cast<float>(flatten_distance_));
			pix++;
		}
	}

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr semantic_pcl;
  semantic_pcl.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
  depthRgba2Pcl(depth_ptr->image, semantic_rgb, semantic_pcl);

  sensor_msgs::PointCloud2 pcl_msgs;
  pcl::toROSMsg(*semantic_pcl, pcl_msgs);
  pcl_msgs.header.stamp = rgb->header.stamp;
  pcl_msgs.header.frame_id = "camera";
  pcl_pub_.publish(pcl_msgs);
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
  label_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", semantic_img)
                  .toImageMsg();
  rgb_msg = cv_bridge::CvImage(std_msgs::Header(), "bgra8", semantic_rgb)
                .toImageMsg();

  image_label_pub_.publish(label_msg);
  image_rgb_pub_.publish(rgb_msg);
  // ROS_INFO("callback_finish");
}

void BisenetRosWrapper::depthRgba2Pcl(
    const cv::Mat &depth, const cv::Mat &rgba,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr semantic_pcl) {
  double width = camera_params_[0];
  double height = camera_params_[1];
  double f = camera_params_[2];

  double center_x = width / 2;
  double center_y = height / 2;

  int rows = depth.rows;
  int cols = depth.cols;
  const float *pix = depth.ptr<float>(0);
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {

      double dist = sqrt(pow(row - center_y, 2) + pow(col - center_x, 2));
      // double dep = depth.at<float>(row, col);
      double dep = *pix;
      double x, y, z;
      uchar r, g, b, a;
      pix++;

			if (maximum_distance_ > 0.0 && dep > maximum_distance_) {
				continue;
			}

      z = dep / sqrt((1 + pow(dist / f, 2)));
      x = z * (col - center_x) / f;
      y = z * (row - center_y) / f;

      cv::Vec4b bgra = rgba.at<cv::Vec4b>(row, col);
      b = bgra[0];
      g = bgra[1];
      r = bgra[2];
      a = bgra[3];

      pcl::PointXYZRGB pt;
      pt.x = x;
      pt.y = y;
      pt.z = z;
      pt.r = r;
      pt.g = g;
      pt.b = b;
      semantic_pcl->push_back(pt);

    }
  }
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

void BisenetRosWrapper::loadColorMap() {
  std::ifstream file(color_file_.c_str());
  CHECK(file.good()) << "Couldn't open file: " << color_file_.c_str();
  std::size_t row_number = 1;
  for (CSVIterator loop(file); loop != CSVIterator(); ++loop) {
    CHECK_EQ(loop->size(), 6) << "Row " << row_number << " is invalid.";

    uint8_t r = std::atoi((*loop)[1].c_str());
    uint8_t g = std::atoi((*loop)[2].c_str());
    uint8_t b = std::atoi((*loop)[3].c_str());
    uint8_t a = std::atoi((*loop)[4].c_str());
    uint8_t id = std::atoi((*loop)[5].c_str());
    color_[id][0] = r;
    color_[id][1] = g;
    color_[id][2] = b;
    color_[id][3] = a;
    row_number++;
  }
}
