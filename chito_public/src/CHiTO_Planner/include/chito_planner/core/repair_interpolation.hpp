#pragma once
#include <Eigen/Core>
#include <vector>
#include <algorithm>
#include <stdexcept>
namespace chito_planner::core {
inline int repair_insertion_budget(int requested,int old_size,int already_inserted,int limit) {
  return std::max(0,std::min(requested,limit-old_size-already_inserted));
}
template<class Joint> std::vector<Joint> repair_interpolation(const Joint& a,const Joint& b,int count) {
  if(count<1||count>31)throw std::invalid_argument("repair_interior_points must be in [1,31]");
  std::vector<Joint> result;result.reserve(count+2);result.push_back(a);
  for(int j=1;j<=count;++j){const double alpha=double(j)/(count+1);result.push_back((1-alpha)*a+alpha*b);}
  result.push_back(b);return result;
}
}
