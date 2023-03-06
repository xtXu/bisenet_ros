#include <iostream>
#include <memory>

#include <torch/script.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(1, 0, 5) << std::endl;
}
