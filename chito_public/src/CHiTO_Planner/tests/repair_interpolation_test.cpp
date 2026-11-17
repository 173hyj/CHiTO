#include <cassert>
#include <iostream>
#if __has_include("chito_planner/core/repair_interpolation.hpp")
#include "chito_planner/core/repair_interpolation.hpp"
using namespace chito_planner::core;
int main(){
 Eigen::Matrix<double,6,1> a,b;a<<0,1,2,3,4,5;b<<6,7,8,9,10,11;
 for(int count:{1,3,5}){
  auto p=repair_interpolation(a,b,count);
  assert(int(p.size())==count+2 && p.front().isApprox(a) && p.back().isApprox(b));
  for(int j=1;j<=count;++j){double t=double(j)/(count+1);assert(p[j].isApprox(((1-t)*a+t*b).eval()));}
 }
 assert(repair_interpolation(a,b,1)[1].isApprox((.5*(a+b)).eval()));
 assert(repair_insertion_budget(5,10,0,256)==5);
 assert(repair_insertion_budget(5,250,4,256)==2);
 assert(repair_insertion_budget(5,256,0,256)==0);
 assert(repair_insertion_budget(1,10,2,256)==1);
 std::cout<<"Interpolation and waypoint-budget tests passed\n";
}
#else
int main(){std::cerr<<"Missing configurable multipoint repair\n";assert(false);}
#endif
