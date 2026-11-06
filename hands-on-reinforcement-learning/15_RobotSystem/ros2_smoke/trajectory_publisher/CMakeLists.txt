cmake_minimum_required(VERSION 3.16)
project(panda_trajectory_smoke)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_executable(publish_trajectory publish_trajectory.cpp)
target_compile_features(publish_trajectory PRIVATE cxx_std_17)
target_link_libraries(publish_trajectory PRIVATE
  rclcpp::rclcpp
  ${trajectory_msgs_TARGETS}
)
install(TARGETS publish_trajectory DESTINATION lib/${PROJECT_NAME})

ament_package()
