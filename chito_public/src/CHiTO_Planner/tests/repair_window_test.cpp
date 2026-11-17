#include "chito_planner/core/repair_window.hpp"
#include <iostream>
#include <limits>
using namespace chito_planner::core;
int main(){int failures=0;auto check=[&](bool ok,const char* label){if(!ok){std::cerr<<"FAIL "<<label<<'\n';++failures;}};
 auto eq=[&](std::vector<double> d,int center,int l,int r,const char* label){auto w=collision_free_window(d,center);check(w.valid&&w.left==l&&w.right==r,label);};
 eq({.1,-.02,.1,.1},2,0,3,"left collision becomes variable");
 eq({.1,.1,-.02,.1},1,0,3,"right collision becomes variable");
 eq({.1,-.02,-.03,-.01,.1},2,0,4,"both colliding sides become variables");
 eq({.1,.1,.1},1,0,2,"safe anchors remain fixed");
 eq({.1,-.02,0.,-.01,.1},2,0,4,"touching not penetrating center");
 check(!collision_free_window({-.1,.1,.1},1).valid,"global start collision not silently fixed");
 check(!collision_free_window({.1,.1,-.1},1).valid,"global goal collision not silently fixed");
 check(!collision_free_window({.1,.1,.1},0).valid,"global endpoint never selected as center");
 check(!collision_free_window({.1,std::numeric_limits<double>::quiet_NaN(),.1},1).valid,"nonfinite invalid");
 check(window_step_acceptable({-.02,-.03,-.04,.01,.02},{-.01,-.02,-.03,.01,.02},2,.005,1.,1.1),"worst improvement beats smoothness");
 check(!window_step_acceptable({-.02,.01,-.03,.01,.02},{-.01,-.001,-.02,.01,.02},2,.005,1.,.9),"point floor protects each positive point to target");
 check(!window_step_acceptable({-.02,.004,-.03,.01,.02},{-.01,.003,-.02,.01,.02},2,.005,1.,.9),"preserve insufficient but positive gap");
 check(!window_step_acceptable({-.02,-.01,-.03,.01,.02},{-.021,-.005,-.02,.01,.02},2,.005,1.,.9),"each penetrating point nonregression");
 check(!window_step_acceptable({.01,.01,.01,.01,.01},{.01,.01,-.001,.01,.01},2,.005,1.,.9),"safe edge cannot become colliding");
 check(!window_step_acceptable({.01,.01,.01,.01,.01},{.01,.01,.01,.01,.01},2,.005,1.,1.1),"feasible merit increase rejected");
 check(window_step_acceptable({.01,.01,.01,.01,.01},{.01,.01,.01,.01,.01},2,.005,1.,.9),"feasible descent accepted");
 check(!window_step_acceptable({.01},{.01},1,.005,1.,.9),"invalid vector dimension rejected");
 std::cout<<"Window test failures="<<failures<<'\n';return failures?1:0;
}
