cmake_minimum_required(VERSION 3.18)
project(llog)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-fPIC")

include_directories(include)

add_library(llog src/llog.cpp)
