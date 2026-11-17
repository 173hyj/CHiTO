#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
namespace chito_planner::core {
struct RepairWindow {int left{-1},right{-1};bool valid{false};};
inline RepairWindow collision_free_window(const std::vector<double>& d,int center){
 if(center<=0||center+1>=int(d.size()))return {};
 for(double v:d)if(!std::isfinite(v))return {};
 if(d.front()<0.||d.back()<0.)return {};
 int left=center-1,right=center+1;
 while(left>0&&d[left]<0.)--left;
 while(right+1<int(d.size())&&d[right]<0.)++right;
 return {left,right,d[left]>=0.&&d[right]>=0.};
}
// First point_count entries are movable-node point distances, the rest edge hulls.
inline bool window_step_acceptable(const std::vector<double>& before,const std::vector<double>& after,int point_count,double target,double merit_before,double merit_after){
 constexpr double tolerance=1e-10;
 if(point_count<1||before.size()!=after.size()||int(before.size())!=2*point_count+1||!std::isfinite(target)||target<0.||!std::isfinite(merit_before)||!std::isfinite(merit_after))return false;
 for(size_t k=0;k<before.size();++k){
  if(!std::isfinite(before[k])||!std::isfinite(after[k]))return false;
  if(before[k]>=0.&&after[k]<std::min(before[k],target)-tolerance)return false;
  if(int(k)<point_count&&after[k]<std::min(before[k],0.)-tolerance)return false;
 }
 double a=std::max(0.,target-*std::min_element(before.begin(),before.end()));
 double b=std::max(0.,target-*std::min_element(after.begin(),after.end()));
 return a>tolerance?b<a-tolerance:b<=tolerance&&merit_after<merit_before-1e-12;
}
}
