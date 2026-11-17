#include "chito_planner/core/split_corridor.hpp"
#include <iostream>
using namespace chito_planner::core;
int main(){int failures=0;auto check=[&](bool v,const char* n){if(!v){std::cerr<<"FAIL "<<n<<'\n';++failures;}};
 auto m=split_corridor_layout({0,0,0,1,1},256);
 check(m.valid&&m.source==std::vector<int>({0,1,1,2,2,3,3,4}),"shared nodes have ordered incoming outgoing copies");
 check(m.cells==std::vector<int>({0,0,0,0,0,1,1,1}),"copies inherit own original cell");
 check(m.connectors==std::vector<int>({1,3,5}),"three within node connections tracked");
 check(m.pieces==std::vector<int>({0,2,4,6}),"original edge pieces remain independent");
 auto two=split_corridor_layout({0,1},2);check(two.valid&&two.source==std::vector<int>({0,1})&&two.connectors.empty(),"true endpoints remain single");
 check(!split_corridor_layout({0,1,2},3).valid,"structural copies count against waypoint budget");
 check(!split_corridor_layout({0,-1,1},20).valid&&!split_corridor_layout({},20).valid,"invalid guide rejected");
 check(split_piece_step({-.02,.01,-.03},{-.01,.01,-.02},2,.005,1.,1.1),"two point one edge infeasible improvement allowed");
 check(!split_piece_step({-.02,.01,-.03},{-.021,.01,-.02},2,.005,1.,.9),"each penetrating point nonregression");
 check(!split_piece_step({.01,.01,.01},{.01,.01,-.001},2,.005,1.,.9),"safe piece edge protected");
 check(!split_piece_step({.01,.01,.01},{.01,.01,.01},2,.005,1.,1.1),"feasible merit increase rejected");
 check(split_piece_step({.01,.01,.01},{.01,.01,.01},2,.005,1.,.9),"feasible merit descent accepted");
 check(!split_piece_step({.01,.01},{.01,.01},2,.005,1.,.9),"missing edge dimension rejected");
 std::cout<<"Split corridor test failures="<<failures<<'\n';return failures?1:0;
}
