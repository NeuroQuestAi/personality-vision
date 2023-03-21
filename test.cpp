#include <iostream>
#include <torch/torch.h>
#include <torchvision/vision.h>

int main(){
  torch::manual_seed(0);
  torch::Tensor x = torch::randn({2,3});
  std::cout << x;
}
