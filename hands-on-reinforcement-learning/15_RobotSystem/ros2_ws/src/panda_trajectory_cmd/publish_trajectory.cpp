#include <chrono>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/empty.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

int main(int argc, char ** argv)
{
  using namespace std::chrono_literals;
  using trajectory_msgs::msg::JointTrajectory;
  using trajectory_msgs::msg::JointTrajectoryPoint;

  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("panda_trajectory_cmd");
  const auto required_subscribers = node->declare_parameter<int>("required_subscribers", 1);
  if (required_subscribers < 1) {
    RCLCPP_ERROR(node->get_logger(), "required_subscribers must be positive");
    rclcpp::shutdown();
    return 1;
  }
  const auto trajectory_case = node->declare_parameter<std::string>("trajectory_case", "single_axis");
  JointTrajectory command;
  command.joint_names = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};
  JointTrajectoryPoint start;
  start.positions = {0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853};
  command.points.push_back(start);
  auto add_point = [&](std::chrono::milliseconds time, std::vector<double> positions) {
    JointTrajectoryPoint point;
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time);
    point.time_from_start.sec = seconds.count();
    point.time_from_start.nanosec =
      std::chrono::duration_cast<std::chrono::nanoseconds>(time - seconds).count();
    point.positions = std::move(positions);
    command.points.push_back(std::move(point));
  };
  std::chrono::milliseconds trajectory_duration;
  if (trajectory_case == "single_axis") {
    trajectory_duration = 400ms;
    add_point(trajectory_duration, {0.1, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853});
  } else if (trajectory_case == "multi_axis") {
    trajectory_duration = 800ms;
    add_point(trajectory_duration, {0.2, -0.18, 0.15, -1.75, 0.12, 1.7, -0.65});
  } else if (trajectory_case == "multi_waypoint") {
    trajectory_duration = 1200ms;
    add_point(400ms, {-0.18, 0.12, -0.10, -1.4, -0.08, 1.45, -0.9});
    add_point(800ms, {0.12, -0.15, 0.15, -1.7, 0.10, 1.72, -0.7});
    add_point(trajectory_duration, {0.0, -0.03, 0.05, -1.5, 0.0, 1.6, -0.8});
  } else {
    RCLCPP_ERROR(node->get_logger(), "Unknown trajectory_case: %s", trajectory_case.c_str());
    rclcpp::shutdown();
    return 1;
  }

  auto publisher = node->create_publisher<JointTrajectory>("trajectory", 10);
  auto heartbeat = node->create_publisher<std_msgs::msg::Empty>(
    "trajectory_heartbeat", rclcpp::QoS(1).best_effort());

  // When recording, both the simulator and recorder must match each writer.
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  const auto count = static_cast<std::size_t>(required_subscribers);
  auto ready = [&]() {
    return publisher->get_subscription_count() >= count &&
           heartbeat->get_subscription_count() >= count;
  };
  while (rclcpp::ok() && !ready() &&
    std::chrono::steady_clock::now() < deadline)
  {
    std::this_thread::sleep_for(20ms);
  }
  if (!rclcpp::ok()) {
    return 0;
  }
  if (!ready()) {
    RCLCPP_ERROR(node->get_logger(), "Expected %zu subscribers on trajectory and heartbeat", count);
    rclcpp::shutdown();
    return 1;
  }

  publisher->publish(command);
  const auto heartbeat_until = std::chrono::steady_clock::now() + trajectory_duration + 200ms;
  while (rclcpp::ok() && std::chrono::steady_clock::now() < heartbeat_until) {
    heartbeat->publish(std_msgs::msg::Empty{});
    std::this_thread::sleep_for(50ms);
  }
  const bool acknowledged = rclcpp::ok() && publisher->wait_for_all_acked(3s);
  if (!rclcpp::ok()) {
    return 0;
  }
  if (acknowledged) {
    RCLCPP_INFO(node->get_logger(), "Published Panda trajectory %s and heartbeat", trajectory_case.c_str());
  } else {
    RCLCPP_ERROR(node->get_logger(), "Timed out waiting for trajectory acknowledgment");
  }
  rclcpp::shutdown();
  return acknowledged ? 0 : 1;
}
