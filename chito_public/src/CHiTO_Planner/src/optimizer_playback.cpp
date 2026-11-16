#include "chito_planner/playback.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
namespace chito_planner {
void export_playback(const OptimizerRequest& r,const OptimizerResult& result,const std::string& path) {
 std::ifstream scene(r.scene_path);if(!scene) throw std::runtime_error("Cannot read playback scene");
 std::ostringstream data;data<<std::setprecision(17)<<"{\"success\":"<<(result.success?"true":"false")<<",\"base\":["<<r.base_xyz.x()<<','<<r.base_xyz.y()<<','<<r.base_xyz.z()<<"],\"yaw\":"<<r.base_yaw_deg<<",\"seed\":";
 auto joints=[&](const JointPath& p){data<<'[';for(size_t i=0;i<p.size();++i){if(i)data<<',';data<<'[';for(int j=0;j<6;++j){if(j)data<<',';data<<p[i][j];}data<<']';}data<<']';};
 joints(result.generation.seed);data<<",\"optimized\":";joints(result.refinement.trajectory);data<<",\"obstacles\":[";
 std::string line;bool first=true;
 while(std::getline(scene,line)) if(line.rfind("*",0)==0){
  double x,y,z,qx,qy,qz,qw;int count;scene>>x>>y>>z>>qx>>qy>>qz>>qw>>count;
  for(int k=0;k<count;++k){std::string type;scene>>type;double a=0,b=0,c=0;if(type=="box")scene>>a>>b>>c;else if(type=="sphere")scene>>a;else throw std::runtime_error("Unsupported playback scene primitive");
   double lx,ly,lz,ax,ay,az,aw;scene>>lx>>ly>>lz>>ax>>ay>>az>>aw;
   std::getline(scene,line);std::getline(scene,line);std::getline(scene,line);
   if(!scene)throw std::runtime_error("Malformed playback scene");if(!first)data<<',';first=false;
   data<<"{\"box\":"<<(type=="box"?"true":"false")<<",\"size\":["<<a<<','<<b<<','<<c<<"],\"p\":["<<x<<','<<y<<','<<z<<"],\"q\":["<<qx<<','<<qy<<','<<qz<<','<<qw<<"],\"lp\":["<<lx<<','<<ly<<','<<lz<<"],\"lq\":["<<ax<<','<<ay<<','<<az<<','<<aw<<"]}";
  }
 }
 if(first)throw std::runtime_error("Playback requires .scene box/sphere geometry; none parsed");
 data<<"]}";std::ofstream out(path);if(!out)throw std::runtime_error("Cannot create playback HTML");
 out<<R"HTML(<!doctype html><meta charset="utf-8"><title>CHiTO schematic playback — not collision certification</title><style>body{font:16px system-ui;background:#f3f5fa;color:#172337;margin:30px}canvas{background:white;width:100%;height:600px;border:1px solid #ccd}input{width:60%}button,select{padding:8px;margin:8px}</style><h2>CHiTO schematic playback — not collision certification</h2><p id="status"></p><button id="play">Play / pause</button><select id="view"><option>iso</option><option>xy</option><option>xz</option></select><input id="time" type="range" min="0" max="1000" value="0"><canvas id="canvas"></canvas><p>Orange dashed: seed. Blue: optimized. Link centerlines use joint-interpolated DH kinematics; these are not collision meshes.</p><script>const D=)HTML"<<data.str()<<R"HTML(;
const cv=document.getElementById('canvas'),slider=document.getElementById('time'),view=document.getElementById('view');document.getElementById('status').textContent=D.success?'Planning succeeded':'Planning failed — available partial paths only';
const mul=(a,b)=>a.map((row,i)=>b[0].map((_,j)=>row.reduce((s,v,k)=>s+v*b[k][j],0)));
function fk(q){let y=D.yaw*Math.PI/180,T=[[Math.cos(y),-Math.sin(y),0,D.base[0]],[Math.sin(y),Math.cos(y),0,D.base[1]],[0,0,1,D.base[2]],[0,0,0,1]],p=[D.base];let a=[0,-.425,-.39225,0,0,0],d=[.089159,0,0,.10915,.09465,.0823],al=[Math.PI/2,0,0,Math.PI/2,-Math.PI/2,0];for(let i=0;i<6;i++){let c=Math.cos(q[i]),s=Math.sin(q[i]),u=Math.cos(al[i]),v=Math.sin(al[i]);T=mul(T,[[c,-s*u,s*v,a[i]*c],[s,c*u,-c*v,a[i]*s],[0,v,u,d[i]],[0,0,0,1]]);p.push(T.slice(0,3).map(r=>r[3]));}return p;}
function rot(p,q){let [x,y,z,w]=q,n=Math.hypot(...q);x/=n;y/=n;z/=n;w/=n;let t=[2*(y*p[2]-z*p[1]),2*(z*p[0]-x*p[2]),2*(x*p[1]-y*p[0])];return p.map((v,i)=>v+w*t[i]+[y*t[2]-z*t[1],z*t[0]-x*t[2],x*t[1]-y*t[0]][i]);}
const add=(a,b)=>a.map((v,i)=>v+b[i]),world=(p,o)=>add(rot(add(rot(p,o.lq),o.lp),o.q),o.p);let wires=[];for(let o of D.obstacles){if(o.box){let p=Array.from({length:8},(_,i)=>world(o.size.map((v,k)=>v*((i>>k&1)-.5)),o));for(let i=0;i<8;i++)for(let j=0;j<3;j++)if(!(i>>j&1))wires.push([p[i],p[i|1<<j]]);}else for(let axis=0;axis<3;axis++){let p=[];for(let i=0;i<=64;i++){let v=[0,0,0];v[(axis+1)%3]=o.size[0]*Math.cos(i*Math.PI/32);v[(axis+2)%3]=o.size[0]*Math.sin(i*Math.PI/32);p.push(world(v,o));}wires.push(p);}}
let paths=[D.seed.map(fk),D.optimized.map(fk)];function project(p){return view.value==='xy'?[p[0],-p[1]]:view.value==='xz'?[p[0],-p[2]]:[p[0]-.65*p[1],-p[2]-.25*p[0]-.25*p[1]];}
function draw(){let w=cv.clientWidth,h=600;cv.width=w*2;cv.height=h*2;let c=cv.getContext('2d');c.scale(2,2);let points=[...wires.flat(),...paths.flat(2),D.base].map(project),lo=[0,1].map(k=>Math.min(...points.map(p=>p[k]))),hi=[0,1].map(k=>Math.max(...points.map(p=>p[k]))),scale=Math.min((w-60)/(hi[0]-lo[0]||1),(h-60)/(hi[1]-lo[1]||1));function line(p,color,width,dash=[]){c.beginPath();p.map(project).forEach((v,i)=>{let xy=[w/2+(v[0]-(lo[0]+hi[0])/2)*scale,h/2+(v[1]-(lo[1]+hi[1])/2)*scale];i?c.lineTo(...xy):c.moveTo(...xy)});c.strokeStyle=color;c.lineWidth=width;c.setLineDash(dash);c.stroke();}wires.forEach(p=>line(p,'#939baa',1));paths.forEach((p,i)=>line(p.map(v=>v[6]),i?'#1261b5':'#d78320',2,i?[]:[5,5]));let p=D.optimized.length?D.optimized:D.seed;if(p.length){let t=+slider.value/1000*(p.length-1),i=Math.floor(t),q=p[i].map((v,k)=>v+(p[Math.min(i+1,p.length-1)][k]-v)*(t-i));line(fk(q),'#243044',5);}}
let playing=false;document.getElementById('play').onclick=()=>playing=!playing;slider.oninput=draw;view.onchange=draw;new ResizeObserver(draw).observe(cv);setInterval(()=>{if(playing){slider.value=(+slider.value+3)%1001;draw();}},30);draw();</script>)HTML";
 out.flush();if(!out)throw std::runtime_error("Failed writing playback HTML");
}
}
