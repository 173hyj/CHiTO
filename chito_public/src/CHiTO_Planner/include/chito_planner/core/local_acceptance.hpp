#pragma once
#include <array>
#include <algorithm>
#include <cmath>
namespace chito_planner::core {
using RepairClearances=std::array<double,3>; // point, left hull, right hull
inline bool local_repair_acceptable(const RepairClearances& before,const RepairClearances& after,double target,double point_floor,double merit_before,double merit_after,bool feasibility_first){
 const auto gaps=[&](const RepairClearances& d){double sum=0.;for(int k=1;k<3;++k)sum+=std::pow(std::max(0.,target-d[k]),2);return sum;};
 if(!feasibility_first)return merit_after<merit_before-1e-12&&gaps(after)<=gaps(before)+1e-12&&after[0]>=std::min(point_floor,before[0])-1e-10;
 constexpr double tolerance=1e-10;
 if(!std::isfinite(target)||target<0.||!std::isfinite(point_floor)||!std::isfinite(merit_before)||!std::isfinite(merit_after))return false;
 for(int k=0;k<3;++k){
  if(!std::isfinite(before[k])||!std::isfinite(after[k]))return false;
  // Preserve nonpenetrating components, capped at the requested target.
  if(before[k]>=0.&&after[k]<std::min(before[k],target)-tolerance)return false;
 }
 if(after[0]<std::min(point_floor,before[0])-tolerance)return false;
 const double worst_before=std::max(0.,target-*std::min_element(before.begin(),before.end()));
 const double worst_after=std::max(0.,target-*std::min_element(after.begin(),after.end()));
 if(worst_before>tolerance)return worst_after<worst_before-tolerance;
 return worst_after<=tolerance&&merit_after<merit_before-1e-12;
}
}
