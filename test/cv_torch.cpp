#include <cmath>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <torch/script.h>

#include <eigen3/Eigen/Eigen>

Eigen::Vector3f mean_ = {0.3257, 0.3690, 0.3223};
Eigen::Vector3f std_ = {0.2112, 0.2148, 0.2115};

int main(int argc, char *argv[]) {

  // load image
  cv::Mat image;
  image =
      cv::imread("/home/xxt/s_explore_ws/src/bisenet_ros/test/img/example.png");

  // load torch module
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(
        "/home/xxt/s_explore_ws/src/bisenet_ros/model/bisenet/model.pt");
    module.eval(); // eval mode
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
	torch::Device device = torch::kCUDA;
	module.to(device);

  // log the original image size
  Eigen::Vector2i org_size = {image.size[0], image.size[1]};
  // make the new size proper for torch
  Eigen::Vector2i new_size;
  new_size[0] = std::ceil(org_size[0] / 32.0) * 32;
  new_size[1] = std::ceil(org_size[1] / 32.0) * 32;

  cv::resize(image, image, cv::Size(new_size[1], new_size[0]));
  // cv: BGR  torch: RGB
  cv::cvtColor(image, image, CV_BGR2RGB);

  // Creat tensor from the cv::Mat
  // cv: H*W*C -> torch: C*H*W
  // use .clone(), to copy the tensor data
  // otherwise, change the vector data will change the tensor
  torch::Tensor tensor_img =
      torch::from_blob(image.data, {new_size[0], new_size[1], 3}, torch::kByte)
          .clone();
  tensor_img = tensor_img.permute({2, 0, 1});
  tensor_img = tensor_img.toType(torch::kFloat);

  // normalize to [0,1]
  tensor_img = tensor_img.div_(255);

  // torch::Tensor mean_tensor = torch::mean(tensor_img, {2,1}, true);
  // torch::Tensor std_tensor = torch::std(tensor_img, {2,1}, true, true);

  // z-score normalize
  torch::Tensor mean_tensor =
      torch::from_blob(mean_.data(), {3, 1, 1}, torch::kFloat).clone();
  torch::Tensor std_tensor =
      torch::from_blob(std_.data(), {3, 1, 1}, torch::kFloat).clone();
  tensor_img = tensor_img.sub_(mean_tensor).div_(std_tensor);

  // add a dimension for batch
  // because torch need B*C*H*W
  tensor_img = tensor_img.unsqueeze(0);
	tensor_img = tensor_img.to(device);

  // creat a torch module input
  std::vector<torch::jit::IValue> tensor_input;
  tensor_input.push_back(tensor_img);

  // inference
  at::Tensor tensor_out = module.forward({tensor_img}).toTensor();
  std::cout << "inference ok \n";

  tensor_out = tensor_out.squeeze(); // delete the batch dimension
  tensor_out = tensor_out.detach();
  tensor_out = tensor_out.to(torch::kU8); // convert the type to uchar8
	tensor_out = tensor_out.to(torch::kCPU);

  cv::Mat semantic_res(cv::Size(org_size[1], org_size[0]), CV_8U,
                       tensor_out.data_ptr());

  // generate a color table, mapping from 0-255 to a specific RGB color
  srand(123);
  uchar color[256][3];
  for (auto &a : color) {
    for (auto &b : a) {
      b = rand() % 256;
    }
  }

  // assign the color to the semantic label,
  // generate the data for generating the semantic iamge
  uchar semantnc_img_data[org_size[0]][org_size[1]][3];
  for (int h = 0; h < org_size[0]; h++) {
    for (int w = 0; w < org_size[1]; w++) {
      uchar label = semantic_res.at<uchar>(h, w);
      semantnc_img_data[h][w][0] = color[label][0];
      semantnc_img_data[h][w][1] = color[label][1];
      semantnc_img_data[h][w][2] = color[label][2];
    }
  }
  cv::Mat semantic_img(cv::Size(org_size[1], org_size[0]), CV_8UC3,
                       semantnc_img_data);

  // save the semantic gray image and RGB image
  cv::imwrite("/home/xxt/s_explore_ws/src/bisenet_ros/test/img/res.png",
              semantic_res);
  cv::imwrite("/home/xxt/s_explore_ws/src/bisenet_ros/test/img/res_color.png",
              semantic_img);

  return 0;
}
