// Deterministic teaching examples. Reported experimental results are separate constants.
export const C={ink:'#272522',muted:'#7b827e',seal:'#6b2717',gold:'#cc9e4c',dun:'#e0d0b6',cadet:'#8b9ea5',green:'#3f7658',red:'#b35249',line:'#e2e6e3',cool:'#edf2f3',pale:'#f5efe5'};
export const clamp=(x,a=0,b=1)=>Math.min(b,Math.max(a,x));
export const mix=(a,b,t)=>a+(b-a)*t;
export const ease=x=>{const t=clamp(x);return t*t*(3-2*t);};
export const phase=(t,a,b)=>ease((t-a)/(b-a));
export const norm=v=>Math.hypot(...v);
export const qMeasured=[-.42,1.28,.45,-.35,.45,-.2];
export const zTarget=[.55,-.45,.8,-.4,.55,-.2];
export const bounded=z=>z.map(x=>.18*Math.tanh(x));
export function forward(q,base=[200,400],lengths=[105,98,74,35,24,18]){
  const points=[base];let angle=-Math.PI/2;
  q.forEach((a,i)=>{angle+=a;const p=points.at(-1);points.push([p[0]+lengths[i]*Math.cos(angle),p[1]+lengths[i]*Math.sin(angle)]);});return points;
}
export function spatial(progress,conflict=false){
  const delta=bounded(zTarget.map(z=>z*ease(progress)));
  const a=delta.map((x,i)=>x*(1+.05*Math.sin(i+1)));
  const b=delta.map((x,i)=>conflict?-x*.88:x*(1-.07*Math.cos(i+1)));
  const d=norm(a.map((x,i)=>x-b[i]))/Math.max(norm(a)+norm(b),1e-8);
  const r=.5*ease(progress),pass=r>=0&&d<=.42671;
  const proposal=qMeasured.map((x,i)=>x+delta[i]);
  return {delta,a,b,d,r,pass,proposal,output:pass?proposal:[...qMeasured]};
}
export function signal(t){return Math.exp(-(((t-1.1)/.28)**2))+.65*Math.exp(-(((t-2.35)/.48)**2))+.3*Math.exp(-(((t-3.3)/.15)**2));}
export function correlation(lag,shift){
  const a=[],b=[];for(let i=0;i<100;i++){const t=.8+i*.022;a.push(signal(t));b.push(signal(t+lag-shift));}
  const ma=a.reduce((x,y)=>x+y)/a.length,mb=b.reduce((x,y)=>x+y)/b.length;
  let top=0,aa=0,bb=0;a.forEach((x,i)=>{const da=x-ma,db=b[i]-mb;top+=da*db;aa+=da*da;bb+=db*db;});return top/Math.sqrt(aa*bb||1);
}
export function lagSearch(shift){const values=Array.from({length:161},(_,i)=>({lag:(i-80)*.01,score:correlation((i-80)*.01,shift)}));return {values,best:values.reduce((a,b)=>a.score>b.score?a:b)};}
export function locate(scenes,time){const index=Math.max(0,scenes.findIndex(s=>time<s.end));const i=time>=scenes.at(-1).end?scenes.length-1:index;return {index:i,local:clamp(time-scenes[i].start,0,scenes[i].end-scenes[i].start)};}
export function cueAt(scene,time){return scene.cues.find(c=>time>=c.start&&time<c.end)||[...scene.cues].reverse().find(c=>time>=c.start)||scene.cues[0];}
export function visualProgress(scene,time){
  const keys=scene.keyframes||[{audio:0,visual:0},{audio:scene.end-scene.start,visual:1}];
  for(let i=1;i<keys.length;i++)if(time<=keys[i].audio){const a=keys[i-1],b=keys[i];return mix(a.visual,b.visual,clamp((time-a.audio)/(b.audio-a.audio||1)));}
  return keys.at(-1).visual;
}
export const RESULTS={franka:{measured:.590,ours:.322,helpful:12,conditions:48},calibration:{before:42.86,after:80.95,retainedBefore:11,retainedAfter:16,total:18},replay:{before:84.06,after:2.26},online:{error:3.07,harmful:.36,recovery:83.05},policy:{raw:58.33,synced:71.67,episodes:240}};
