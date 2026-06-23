cmake_minimum_required(VERSION 3.18)
project(kaolin_wisp_cpp)

find_package(Torch REQUIRED)

# kaolin refer: kaolin/setup.py
add_definitions(-O3 -DWITH_CUDA -DTHRUST_IGNORE_CUB_VERSION_CHECK)
include_directories(submodules/kaolin kaolin_wisp_cpp)

file(
  GLOB_RECURSE
  KAOLIN_SOURCES
  submodules/kaolin/kaolin/csrc/ops/*.cpp
  submodules/kaolin/kaolin/csrc/ops/*.cu
  submodules/kaolin/kaolin/csrc/render/*.cpp
  submodules/kaolin/kaolin/csrc/render/*.cu)

list(
  FILTER
  KAOLIN_SOURCES
  EXCLUDE
  REGEX
  "submodules/kaolin/kaolin/csrc/ops/mesh/triangle_hash.cpp|submodules/kaolin/kaolin/csrc/ops/spc/bf.cpp|submodules/kaolin/kaolin/csrc/ops/spc/bf_cuda.cu|submodules/kaolin/kaolin/csrc/ops/spc/recon.cpp|submodules/kaolin/kaolin/csrc/ops/spc/recon_cuda.cu"
)

add_library(
  kaolin_wisp_cpp
  kaolin_wisp_cpp/spc_ops/spc_ops.cpp
  kaolin_wisp_cpp/wisp_spc_ops/wisp_spc_ops.cpp
  kaolin_wisp_cpp/octree_as/octree_as.cpp
  kaolin_wisp_cpp/render_ops/find_depth_bound.cpp
  kaolin_wisp_cpp/render_ops/find_depth_bound_cuda.cu
  ${KAOLIN_SOURCES})
target_link_libraries(kaolin_wisp_cpp ${TORCH_LIBRARIES})
