// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
// Copyright (c) 2023 Open Navigation LLC
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

#ifndef NAV2_MPPI_CONTROLLER__CRITICS__PREFER_RIGHT_CRITIC_HPP_
#define NAV2_MPPI_CONTROLLER__CRITICS__PREFER_RIGHT_CRITIC_HPP_

#include "nav2_mppi_controller/critic_function.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"

namespace mppi::critics
{

/**
 * @class mppi::critics::ConstraintCritic
 * @brief Critic objective function for following the path approximately
 * To allow for deviation from path in case of dynamic obstacles. Path Align
 * is what aligns the trajectories to the path more or less precisely, if desireable.
 * A higher weight here with an offset > 1 will accelerate the samples to full speed
 * faster and push the follow point further ahead, creating some shortcutting.
 */
class PreferRightCritic : public CriticFunction
{
public:
  /**
    * @brief Initialize critic
    */
  void initialize() override;

  /**
   * @brief Evaluate cost related to robot orientation at goal pose
   * (considered only if robot near last goal in current plan)
   *
   * @param costs [out] add goal angle cost values to this tensor
   */
  void score(CriticData & data) override;

protected:
  float threshold_to_consider_{0};
  size_t offset_from_furthest_{0};

  unsigned int power_{0};
  float weight_{0};
};

}  // namespace mppi::critics

#endif  // NAV2_MPPI_CONTROLLER__CRITICS__PREFER_RIGHT_CRITIC_HPP_
