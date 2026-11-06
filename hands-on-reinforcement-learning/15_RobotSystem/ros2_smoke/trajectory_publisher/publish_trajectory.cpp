#include <chrono>
#include <limits>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

int main(int argc, char ** argv)
{
  using namespace std::chrono_literals;
  using trajectory_msgs::msg::JointTrajectory;
  using trajectory_msgs::msg::JointTrajectoryPoint;

  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("panda_trajectory_smoke");
  auto publisher = node->create_publisher<JointTrajectory>("/panda_smoke/trajectory", 10);

  // Wait for both the simulator and the bag recorder.
  const auto deadline = std::chrono::steady_clock::now() + 15s;
  while (rclcpp::ok() && publisher->get_subscription_count() < 2 &&
    std::chrono::steady_clock::now() < deadline)
  {
    std::this_thread::sleep_for(20ms);
  }
  if (!rclcpp::ok() || publisher->get_subscription_count() < 2) {
    RCLCPP_ERROR(node->get_logger(), "Simulator and recorder did not both match");
    rclcpp::shutdown();
    return 1;
  }

  JointTrajectory command;
  command.joint_names = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};
  JointTrajectoryPoint start;
  start.positions = {0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853};
  auto target = start;
  target.positions[0] = 0.15;
  target.time_from_start.nanosec = 400000000;
  command.points = {start, target};

  auto invalid = command;
  invalid.joint_names[0] = "unknown";
  publisher->publish(invalid);
  invalid = command;
  invalid.points[1].positions[0] = std::numeric_limits<double>::quiet_NaN();
  publisher->publish(invalid);
  invalid = command;
  invalid.points[1].positions[0] = 10.0;
  publisher->publish(invalid);
  invalid = command;
  invalid.points[1].time_from_start = start.time_from_start;
  publisher->publish(invalid);
  publisher->publish(command);

  const bool delivered = publisher->wait_for_all_acked(3s);
  if (delivered) {
    RCLCPP_INFO(node->get_logger(), "Published four fault cases and one fixed trajectory");
  } else {
    RCLCPP_ERROR(node->get_logger(), "Timed out waiting for message acknowledgments");
  }
  rclcpp::shutdown();
  return delivered ? 0 : 1;
}
