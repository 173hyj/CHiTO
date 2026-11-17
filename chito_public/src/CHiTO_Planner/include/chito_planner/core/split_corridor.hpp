#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
namespace chito_planner::core {
struct SplitCorridorLayout{bool valid{false};std::vector<int> source,cells,connectors,pieces;};
inline SplitCorridorLayout split_corridor_layout(const std::vector<int>& cells,int limit){
 SplitCorridorLayout m;if(cells.size()<2||2*cells.size()-2>size_t(std::max(0,limit)))return m;
 for(int id:cells)if(id<0)return m;
 for(int j=0;j<int(cells.size());++j){m.source.push_back(j);m.cells.push_back(cells[j]);if(j>0&&j+1<int(cells.size())){m.connectors.push_back(int(m.source.size())-1);m.source.push_back(j);m.cells.push_back(cells[j]);}}
 for(int j=0;j+1<int(m.source.size());++j)if(m.source[j]!=m.source[j+1])m.pieces.push_back(j);m.valid=true;return m;
}
inline bool split_piece_step(const std::vector<double>& a,const std::vector<double>& b,int points,double target,double before,double after){
 if(points<1||a.size()!=b.size()||a.size()<=size_t(points)||!std::isfinite(target)||target<0||!std::isfinite(before)||!std::isfinite(after))return false;
 for(size_t k=0;k<a.size();++k){if(!std::isfinite(a[k])||!std::isfinite(b[k]))return false;if(a[k]>=0&&b[k]<std::min(a[k],target)-1e-10)return false;if(k<size_t(points)&&b[k]<std::min(a[k],0.)-1e-10)return false;}
 double old=std::max(0.,target-*std::min_element(a.begin(),a.end())),next=std::max(0.,target-*std::min_element(b.begin(),b.end()));return old>1e-10?next<old-1e-10:next<=1e-10&&after<before-1e-12;
}
}
