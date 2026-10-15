cmake_minimum_required(VERSION 3.5)
project(gazebo_grasp_attach)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_library(gazebo_grasp_attach SHARED src/grasp_attach.cpp)
ament_target_dependencies(gazebo_grasp_attach gazebo_dev gazebo_ros rclcpp std_msgs)
ament_export_libraries(gazebo_grasp_attach)
ament_export_dependencies(gazebo_dev gazebo_ros rclcpp std_msgs)

install(TARGETS gazebo_grasp_attach LIBRARY DESTINATION lib)

ament_package()
