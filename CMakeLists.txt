cmake_minimum_required(VERSION 3.0)
project(debugfunc)

set(CMAKE_PREFIX_PATH /home/edmc/torch/install/libtorch)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} main.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_range_for)
target_link_libraries(${PROJECT_NAME} TorchVision::TorchVision)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

