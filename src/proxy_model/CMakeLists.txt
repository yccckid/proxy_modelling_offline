cmake_minimum_required(VERSION 3.0.2)
project(proxy_model)

find_package(catkin REQUIRED)
catkin_python_setup()
catkin_package()

catkin_install_python(
  PROGRAMS scripts/build_object_bag.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
