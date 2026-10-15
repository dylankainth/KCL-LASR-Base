// Gazebo Classic world plugin: attach / detach a model link to another model link.
//
// Gazebo's ODE contact handling of a small object squeezed by parallel-jaw fingers is
// unreliable (the object is ejected or slips), so simulated grasps are assisted by a
// joint between the gripper and the object, created only once the gripper has closed on
// the object (same idea as gazebo_grasp_plugin / gazebo_ros_link_attacher).
//
// Topic  ~/grasp_attach/command  (std_msgs/String):
//   "attach <model1> <link1> <model2> <link2>"   joint with model1::link1 as parent
//   "detach <model1> <link1> <model2> <link2>"
//
// The joint holds the *current* relative pose of the two links.

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

namespace lasr
{

class GraspAttach : public gazebo::WorldPlugin
{
public:
  void Load(gazebo::physics::WorldPtr world, sdf::ElementPtr sdf) override
  {
    world_ = world;
    node_ = gazebo_ros::Node::Get(sdf);
    sub_ = node_->create_subscription<std_msgs::msg::String>(
      "grasp_attach/command", 10,
      [this](std_msgs::msg::String::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(msg->data);
      });
    update_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      [this](const gazebo::common::UpdateInfo &) { this->OnUpdate(); });
    RCLCPP_INFO(node_->get_logger(), "grasp_attach ready on %s", sub_->get_topic_name());
  }

private:
  void OnUpdate()
  {
    std::queue<std::string> pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      std::swap(pending, queue_);
    }
    while (!pending.empty()) {
      Handle(pending.front());
      pending.pop();
    }
  }

  void Handle(const std::string & cmd)
  {
    std::istringstream ss(cmd);
    std::vector<std::string> t;
    for (std::string w; ss >> w;) {
      t.push_back(w);
    }
    if (t.size() != 5) {
      RCLCPP_WARN(node_->get_logger(), "bad command '%s'", cmd.c_str());
      return;
    }
    const std::string key = t[1] + "::" + t[2] + "|" + t[3] + "::" + t[4];
    if (t[0] == "attach") {
      Attach(key, t[1], t[2], t[3], t[4]);
    } else if (t[0] == "detach") {
      Detach(key);
    } else {
      RCLCPP_WARN(node_->get_logger(), "unknown command '%s'", t[0].c_str());
    }
  }

  void Attach(
    const std::string & key, const std::string & model1, const std::string & link1,
    const std::string & model2, const std::string & link2)
  {
    if (joints_.count(key)) {
      return;  // already attached
    }
    auto m1 = world_->ModelByName(model1);
    auto m2 = world_->ModelByName(model2);
    if (!m1 || !m2) {
      RCLCPP_WARN(node_->get_logger(), "attach: model not found (%s / %s)", model1.c_str(), model2.c_str());
      return;
    }
    auto l1 = m1->GetLink(link1);
    auto l2 = m2->GetLink(link2);
    if (!l1 || !l2) {
      RCLCPP_WARN(node_->get_logger(), "attach: link not found (%s / %s)", link1.c_str(), link2.c_str());
      return;
    }
    // A zero-limit revolute joint is the robust way to get a rigid joint in Gazebo 11.
    auto joint = world_->Physics()->CreateJoint("revolute", m1);
    joint->SetName("grasp_attach_" + std::to_string(joints_.size()) + "_" + model2);
    joint->SetModel(m2);
    joint->Load(l1, l2, ignition::math::Pose3d());
    joint->Init();
    joint->SetAxis(0, ignition::math::Vector3d(0, 0, 1));
    joint->SetUpperLimit(0, 0);
    joint->SetLowerLimit(0, 0);
    joints_[key] = joint;
    RCLCPP_INFO(node_->get_logger(), "attached %s", key.c_str());
  }

  void Detach(const std::string & key)
  {
    auto it = joints_.find(key);
    if (it == joints_.end()) {
      return;
    }
    it->second->Detach();
    joints_.erase(it);
    RCLCPP_INFO(node_->get_logger(), "detached %s", key.c_str());
  }

  gazebo::physics::WorldPtr world_;
  gazebo_ros::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
  gazebo::event::ConnectionPtr update_;
  std::mutex mutex_;
  std::queue<std::string> queue_;
  std::map<std::string, gazebo::physics::JointPtr> joints_;
};

GZ_REGISTER_WORLD_PLUGIN(GraspAttach)

}  // namespace lasr
