#include <chrono>
#include <thread>

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
  auto publisher = node->create_publisher<JointTrajectory>("trajectory", 10);
  auto heartbeat = node->create_publisher<std_msgs::msg::Empty>(
    "trajectory_heartbeat", rclcpp::QoS(1).best_effort());

  // Match both readers before starting the trajectory.
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  auto ready = [&]() {
    return publisher->get_subscription_count() && heartbeat->get_subscription_count();
  };
  while (rclcpp::ok() && !ready() &&
    std::chrono::steady_clock::now() < deadline)
  {
    std::this_thread::sleep_for(20ms);
  }
  if (!rclcpp::ok() || !ready()) {
    RCLCPP_ERROR(node->get_logger(), "No matching trajectory or heartbeat subscriber");
    rclcpp::shutdown();
    return 1;
  }

  JointTrajectory command;
  command.joint_names = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};
  JointTrajectoryPoint start;
  start.positions = {0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853};
  auto target = start;
  target.positions[0] = 0.1;
  constexpr auto trajectory_duration = 400ms;
  target.time_from_start.nanosec =
    std::chrono::duration_cast<std::chrono::nanoseconds>(trajectory_duration).count();
  command.points = {start, target};

  publisher->publish(command);
  const auto heartbeat_until = std::chrono::steady_clock::now() + trajectory_duration + 200ms;
  while (rclcpp::ok() && std::chrono::steady_clock::now() < heartbeat_until) {
    heartbeat->publish(std_msgs::msg::Empty{});
    std::this_thread::sleep_for(50ms);
  }
  const bool acknowledged = rclcpp::ok() && publisher->wait_for_all_acked(3s);
  if (acknowledged) {
    RCLCPP_INFO(node->get_logger(), "Published a two-point Panda trajectory and heartbeat");
  } else {
    RCLCPP_ERROR(node->get_logger(), "Timed out waiting for trajectory acknowledgment");
  }
  rclcpp::shutdown();
  return acknowledged ? 0 : 1;
}
