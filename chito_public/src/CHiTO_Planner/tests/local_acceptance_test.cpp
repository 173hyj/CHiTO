#include "chito_planner/core/local_acceptance.hpp"
#include <iostream>
#include <limits>
#include <random>
using namespace chito_planner::core;
int main(){int failures=0;auto check=[&](bool v,const char* label){if(!v){std::cerr<<"FAIL "<<label<<'\n';++failures;}};
 auto accept=[](RepairClearances a,RepairClearances b,double old=1.,double next=.9){return local_repair_acceptable(a,b,.005,0.,old,next,true);};
 check(accept({.02,-.020,-.001},{.02,-.015,-.002},1.,1.1),"worst clearance improves despite smoothness cost");
 check(!accept({.02,-.010,-.009},{.02,-.011,.004}),"sum-gap compensation cannot worsen worst penetration");
 check(!accept({.02,-.020,.006},{.02,-.010,.004}),"preserve achieved safe-side target");
 check(!accept({.02,-.020,.003},{.02,-.010,-.001}),"do not introduce collision on nonpenetrating side");
 check(!accept({.02,-.020,.003},{.02,-.010,.002}),"preserve existing clearance below target on safe side");
 check(!accept({-.01,-.020,.006},{-.011,-.010,.006}),"point nonregression preserved");
 check(!accept({.02,-.020,.006},{.02,-.020,.006}),"smoothness alone cannot accept infeasible stationary step");
 check(accept({.02,.006,.007},{.02,.006,.007}),"feasible merit descent accepted");
 check(!accept({.02,.006,.007},{.02,.006,.007},1.,1.1),"feasible merit increase rejected");
 check(!accept({.02,.006,.007},{.02,.004,.007}),"feasible result must retain target");
 check(!accept({.02,-.02,.007},{.02,-.01,std::numeric_limits<double>::quiet_NaN()}),"nonfinite distances reject");
 check(!accept({.02,-.02,.007},{.02,-.01,.007},1.,std::numeric_limits<double>::infinity()),"nonfinite merit rejects");
 std::mt19937 rng(11);std::uniform_real_distribution<double> d(-.05,.05),m(0.,2.);
 for(int i=0;i<1000;++i){RepairClearances a{d(rng),d(rng),d(rng)},b{d(rng),d(rng),d(rng)};double ma=m(rng),mb=m(rng);auto sq=[](double x){return std::pow(std::max(0.,.005-x),2);};bool expected=mb<ma-1e-12&&sq(b[1])+sq(b[2])<=sq(a[1])+sq(a[2])+1e-12&&b[0]>=std::min(0.,a[0])-1e-10;check(local_repair_acceptable(a,b,.005,0.,ma,mb,false)==expected,"legacy equivalence");}
 std::cout<<"Acceptance regressions failures="<<failures<<'\n';return failures?1:0;
}
