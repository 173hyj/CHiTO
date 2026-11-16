#pragma once
#include "chito_planner/optimizer.hpp"
namespace chito_planner {
OptimizerRequest read_optimizer_config(const std::string& filename);
void validate_optimizer_request(const OptimizerRequest& request);
}
