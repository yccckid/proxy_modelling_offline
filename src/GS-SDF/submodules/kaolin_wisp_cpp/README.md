# kaolin_wisp_cpp

```bash
# LibTroch: https://pytorch.org/get-started/locally/
  # Tested on:
  # - CUDA/Torch: 11.8/2.0.0
  wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
  # for other cuda version use the following command to download the corresponding version of libtorch(cuda11.3 for example):
  # wget https://download.pytorch.org/libtorch/nightly/cu113/libtorch-cxx11-abi-shared-with-deps-latest.zip
  unzip libtorch-cxx11-abi-shared-with-deps-*.zip
  echo "export Torch_DIR=$PWD/libtorch/share/cmake/Torch" >> ~/.zshrc
```

CMakelist.txt
```bash
# kaolin_wisp_cpp
add_subdirectory(submodules/kaolin_wisp_cpp)
include_directories(submodules/kaolin_wisp_cpp
                    submodules/kaolin_wisp_cpp/submodules/kaolin)
```