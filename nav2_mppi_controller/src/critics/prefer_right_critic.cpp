// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "nav2_mppi_controller/critics/prefer_right_critic.hpp"

#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>

namespace mppi::critics
{

void PreferRightCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(
    threshold_to_consider_,
    "threshold_to_consider", 1.0);
  getParam(offset_from_furthest_, "offset_from_furthest", 6);
  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 5.0);
  getParam(radius_, "radius", 0.26);
}

void PreferRightCritic::score(CriticData & data)
{
  if (!enabled_ || data.path.x.shape(0) < 2 ||
    utils::withinPositionGoalTolerance(threshold_to_consider_, data.state.pose.pose, data.path))
  {
    return;
  }

  utils::setPathFurthestPointIfNotSet(data);
  utils::setPathCostsIfNotSet(data, costmap_ros_);
  const size_t path_size = data.path.x.shape(0) - 1;

  auto offseted_idx = std::min(
    *data.furthest_reached_path_point + offset_from_furthest_, path_size);

  // Drive to the first valid path point, in case of dynamic obstacles on path
  // we want to drive past it, not through it
  bool valid = false;
  while (!valid && offseted_idx < path_size - 1) {
    valid = (*data.path_pts_valid)[offseted_idx];
    if (!valid) {
      offseted_idx++;
    }
  }

  // RCLCPP_INFO(
  //   logger_, "PreferRightCritic testing %ld of %ld", offseted_idx, path_size);

  // const auto x0 = data.path.x(0);
  // const auto y0 = data.path.y(0);

  // const auto xf = data.path.x(path_size-1);
  // const auto yf = data.path.y(path_size-1);

  const auto path_x = data.path.x(offseted_idx);
  const auto path_y = data.path.y(offseted_idx);

  // RCLCPP_INFO(
  //  logger_, "PreferRightCritic %f, %f -> [%f, %f] -> %f, %f",
  //  x0, y0, path_x, path_y, xf, yf);

  const auto path_yaw = data.path.yaws(offseted_idx);
  const auto left_dx = -sin(path_yaw);
  const auto left_dy = cos(path_yaw);

  const auto last_x = xt::view(data.trajectories.x, xt::all(), -1);
  const auto last_y = xt::view(data.trajectories.y, xt::all(), -1);

  auto adjustment = xt::eval(radius_ + (last_x - path_x) * left_dx + (last_y - path_y) * left_dy);
  adjustment = xt::minimum(adjustment, 2.0*radius_);
  adjustment = xt::maximum(adjustment, 0.0);

  auto costs = xt::eval(adjustment);

  // RCLCPP_INFO(logger_, "PreferRightCritic %f, %f", xt::amin(costs)(), xt::amax(costs)());

  data.costs += xt::pow(weight_ * std::move(costs), power_);
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::PreferRightCritic,
  mppi::critics::CriticFunction)
