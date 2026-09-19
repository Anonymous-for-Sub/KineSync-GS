import {C,clamp,mix,phase,verificationState,signal,lagSearch} from './tour-math.mjs?v=4.2';

const txt=(x,y,s,size=22,color=C.ink,weight=500,anchor='start')=>`<text x="${x}" y="${y}" fill="${color}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${s}</text>`;
const rect=(x,y,w,h,fill=C.cool,stroke='none',r=7)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}"/>`;
const line=(x,y,u,v,c=C.line,w=2,dash='')=>`<path d="M${x},${y} L${u},${v}" stroke="${c}" stroke-width="${w}" fill="none" ${dash?`stroke-dasharray="${dash}"`:''}/>`;
const circle=(x,y,r,c)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${c}"/>`;
const fade=(s,p)=>`<g opacity="${clamp(p)}">${s}</g>`;
const arrow=(x,y,u,v,p=1,c=C.gold)=>{const t=clamp(p),xx=mix(x,u,t),yy=mix(y,v,t),a=Math.atan2(v-y,u-x);return line(x,y,u,v,C.line,2)+line(x,y,xx,yy,c,3)+`<path d="M-9,-6 L0,0 L-9,6" transform="translate(${xx},${yy}) rotate(${a*180/Math.PI})" fill="none" stroke="${c}" stroke-width="3"/>`;};
const path=(a,c,w=3)=>`<path d="${a.map(([x,y],i)=>`${i?'L':'M'}${x.toFixed(2)},${y.toFixed(2)}`).join(' ')}" stroke="${c}" stroke-width="${w}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>`;
function vector(x,y,values,color,w=230,h=98){let s='';values.forEach((v,i)=>{const dx=x+i*w/values.length;s+=rect(dx,y,w/values.length-9,h,'#f0f3f3')+line(dx,y+h/2,dx+w/values.length-9,y+h/2,'#b8c6c9',1)+rect(dx+6,v>=0?y+h/2-v*h*2:y+h/2,w/values.length-21,Math.abs(v)*h*2,color,'none',2)+txt(dx+12,y+h+27,`q${i+1}`,18,C.muted);});return s;}
const nodes=(x,y,labels,p,w=1120)=>labels.map((label,i)=>{const xx=x+i*w/labels.length;return fade(rect(xx,y,w/labels.length-40,53,i===labels.length-1?'#e6f0e8':'#f0f3f3')+txt(xx+(w/labels.length-40)/2,y+33,label,20,i===labels.length-1?C.green:C.ink,550,'middle'),.3+.7*phase(p,i*.18,i*.18+.2))+(i<labels.length-1?arrow(xx+w/labels.length-33,y+26,xx+w/labels.length-8,y+26,phase(p,i*.18,.2+i*.18)):'');}).join('');
const metric=(x,y,value,label,color=C.seal,size=46)=>txt(x,y,value,size,color,650)+txt(x,y+33,label,20,C.muted);
const packet=(x,y,u,v,p,c=C.gold)=>{const t=clamp(p);return circle(mix(x,u,t),mix(y,v,t),6,c);};
const check=(x,y,c=C.green)=>`<path d="M${x-7},${y} l5,5 l10,-12" stroke="${c}" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>`;

// The SVG, recorded media and interactive robot share this coordinate system.
export const layouts={
 'real2sim2real':{robots:[{x:0,y:55,w:445,h:365},{x:755,y:55,w:445,h:365}],media:[{x:385,y:303,w:430,h:243}],asset:'primary'},
 failure:{robots:[{x:650,y:115,w:505,h:370}],media:[{x:28,y:118,w:550,h:326}],asset:'motivation'},
 contract:{robots:[{x:870,y:133,w:310,h:282}],media:[]},
 spatial:{robots:[{x:15,y:50,w:530,h:410}],media:[{x:630,y:300,w:540,h:230}],asset:'primary'},
 temporal:{robots:[],media:[{x:635,y:255,w:535,h:276}],asset:'primary'},
 controlled:{robots:[],media:[{x:555,y:135,w:615,h:310}],asset:'franka'},
 mechanism:{robots:[],media:[{x:665,y:135,w:500,h:340}],asset:'primary'},
 'real-world':{robots:[],media:[{x:28,y:77,w:570,h:324},{x:665,y:105,w:510,h:280}],asset:'piper'},
 online:{robots:[],media:[]},
 policy:{robots:[],media:[{x:627,y:116,w:540,h:307}],asset:'primary'},
 closing:{robots:[{x:5,y:50,w:470,h:400}],media:[{x:690,y:140,w:485,h:280}],asset:'primary'}
};
export const extraMedia={
 motivation:{media:'static/images/motivation.png',label:'Measured image-fitting and state-error traces'},
 franka:{media:'static/images/franka-clean-diagnostic.png',label:'Controlled Franka measured and verified states'},
 piper:{media:'static/images/piper-alignment.png',label:'Real-world PiPER alignment across matched observations'}
};
export function evidenceFor(scene){const l=layouts[scene.id];if(l.asset==='piper')return [scene.evidence,extraMedia.piper];if(extraMedia[l.asset])return [extraMedia[l.asset]];return scene.evidence?[scene.evidence]:[];}

function opening(p){
 const t=phase(p,.04,.9),q=[.42*Math.sin(t*4.08),.2*Math.sin(t*Math.PI),0,-.26*Math.sin(t*Math.PI),0,.18*Math.sin(t*Math.PI),0];
 let s=txt(57,48,'Articulated geometry',26,C.cadet,600)+txt(826,48,'Gaussian appearance',26,C.seal,600);
 s+=rect(523,94,154,62,'#edf3ef')+txt(600,135,'q(t)',31,C.green,600,'middle');
 s+=arrow(520,127,418,185,phase(p,.06,.24),C.cadet)+arrow(680,127,782,185,phase(p,.06,.24),C.gold);
 s+=packet(520,127,418,185,(p*3)%1,C.cadet)+packet(680,127,782,185,(p*3)%1,C.gold);
 q.forEach((v,i)=>{s+=line(517+i*25,190,517+i*25,232,C.line,4)+line(517+i*25,211,517+i*25,211-v*45,i%2?C.gold:C.cadet,7);});
 return s+txt(600,275,'Real-world observations',23,C.ink,550,'middle')+fade(txt(206,470,'Same articulation',24,C.cadet,550,'middle')+txt(998,470,'Same timestamp',24,C.seal,550,'middle'),phase(p,.55,.85));
}
function failure(p){return txt(35,65,'Observed during inverse rendering',25,C.ink,550)+circle(698,73,7,C.cadet)+txt(717,81,'Current image',23,C.cadet,550)+circle(960,73,7,C.gold)+txt(979,81,'Delayed image',23,C.gold,550)+fade(rect(35,478,530,62,'#f5efea')+txt(300,516,'Image error ↓   Joint error ↑',28,C.seal,600,'middle'),phase(p,.15,.85))+circle(710,519,7,C.cadet)+txt(728,526,'Measured',23,C.cadet,550)+circle(949,519,7,C.gold)+txt(967,526,'Fitted',23,C.seal,550);}
function contract(p,o){
 const v=verificationState(p,o.branch==='conflict'),ev=phase(p,.32,.56),decision=phase(p,.58,.78),color=v.pass?C.green:C.seal;
 let s=txt(35,59,'1  Propose',27,C.gold,600)+txt(445,59,'2  Verify',27,ev>.2?C.seal:C.cadet,600)+txt(914,59,'3  Synchronize',27,v.assessed?color:C.cadet,600);
 s+=txt(35,127,'One bounded joint correction',22)+vector(45,162,v.delta,C.gold,275,110)+txt(45,334,'δ = b tanh(z)',31,C.seal)+txt(45,374,'q⁺ = qᵐ + δ',28)+arrow(340,223,443,223,phase(p,.2,.44));
 s+=`<path d="M590,119 L716,230 L590,341 L464,230 Z" fill="${v.assessed?(v.pass?'#edf4ee':'#f7efeb'):'#f1f4f3'}" stroke="${v.assessed?color:C.cadet}" stroke-width="2"/>`;
 s+=txt(590,210,'Cross-view',25,C.ink,550,'middle')+txt(590,245,'evidence',25,C.ink,550,'middle');
 s+=fade(txt(590,285,`d = ${v.d.toFixed(3)}`,22,v.assessed?color:C.muted,550,'middle'),ev);
 s+=arrow(718,230,883,230,v.pass?decision:0,v.pass?C.green:C.cadet)+fade(txt(802,195,v.pass?'Commit':'Fallback',22,color,600,'middle'),decision);
 s+=txt(1020,438,v.assessed?(v.pass?'qˢ = q⁺':'qˢ = qᵐ'):'qˢ = qᵐ',31,v.assessed?color:C.cadet,600,'middle');
 s+=line(182,407,182,478,C.cadet,2,'6 5')+line(182,478,1020,478,C.cadet,2,'6 5');
 if(v.assessed&&!v.pass)s+=line(182,407,182,478,C.cadet,3)+arrow(182,478,1020,478,decision,C.cadet);
 return s+txt(600,540,v.assessed?(v.pass?'Commit the coordinated correction.':'Retain the complete measured state.'):'Estimation proposes. Evidence authorizes.',26,v.assessed?color:C.ink,550,'middle');
}
function spatialScene(p,o){
 const v=verificationState(p,o.branch==='conflict'),ev=phase(p,.25,.56),color=v.pass?C.green:C.seal;
 let s=circle(88,62,7,C.cadet)+txt(106,70,'View A',23,C.cadet,550)+circle(371,62,7,C.gold)+txt(389,70,'View B',23,C.gold,550);
 s+=circle(80,508,6,C.cadet)+txt(96,516,'Synchronized state',22,C.ink,550)+circle(348,508,6,C.gold)+txt(364,516,'Proposal',22,C.gold,550);
 s+=txt(615,45,'Cross-view support for one proposal',25,C.ink,550)+fade(txt(619,91,'δA',23,C.cadet,600)+vector(661,73,v.a,C.cadet,209,74)+txt(902,91,'δB',23,C.gold,600)+vector(944,73,v.b,C.gold,209,74),.18+.82*ev);
 s+=txt(619,213,'Disagreement',22)+rect(803,197,350,13,'#e9edef')+rect(803,197,350*v.d*ev,13,v.assessed?color:C.gold)+line(803+350*.42671,187,803+350*.42671,222,C.ink,2)+txt(803+350*.42671,250,'τd',20,C.muted,500,'middle');
 s+=fade(txt(1153,250,v.d.toFixed(3),24,color,600,'end'),ev);
 return s+(v.assessed?(v.pass?check(631,270):line(624,263,638,277,C.seal,3)+line(638,263,624,277,C.seal,3)):'')+txt(653,278,v.assessed?(v.pass?'Commit whole state':'Retain measurement'):'Gather cross-view evidence',25,v.assessed?color:C.muted,600);
}
function temporal(p,o){
 const shift=o.lag??.45,aligned=o.aligned??phase(p,.52,.84),residual=shift*(1-aligned),search=lagSearch(shift);
 const scan=phase(p,.08,.5),peak=search.best,peakX=646+(peak.lag+.8)/1.6*487;
 let s=txt(35,54,'Match motion, then resample state',25,C.ink,550);
 for(const t of [1.1,2.35,3.3]){
  const a=45+t/4*505,b=45+(t+residual)/4*505;
  s+=rect(a-9,113,18,85,'#edf2f3')+rect(b-9,252,18,85,aligned>.95?'#e8f1eb':'#f6efdf');
  s+=line(a,199,b,245,aligned>.95?C.green:C.gold,2,'5 4');
 }
 for(const [y,delay,color,label] of [[116,0,C.cadet,'Telemetry'],[255,residual,C.gold,'Image motion']]){
  s+=txt(35,y-23,label,22,color,600)+line(45,y+78,550,y+78,C.line,2)+path(Array.from({length:100},(_,i)=>{const t=i/99*4;return [45+i/99*505,y+76-signal(t-delay)*76];}),color,4);
  for(const t of [1.1,2.35,3.3])s+=circle(45+(t+delay)/4*505,y+76-signal(t)*76,5,color);
 }
 const selected=search.values[Math.min(160,Math.floor(scan*160))];
 s+=txt(633,53,'Search for shared motion',25,C.ink,550)+line(646,205,1133,205,C.line,2);
 s+=path(search.values.slice(0,Math.max(2,Math.ceil(scan*161))).map(v=>[646+(v.lag+.8)/1.6*487,200-(v.score+1)*67]),C.seal,3);
 if(scan<1)s+=circle(646+(selected.lag+.8)/1.6*487,200-(selected.score+1)*67,7,C.gold);
 s+=fade(line(peakX,67,peakX,205,C.green,2,'5 4')+circle(peakX,200-(peak.score+1)*67,6,C.green)+txt(1148,236,`Δt⁺ = ${Math.round(peak.lag*1000)} ms`,24,C.seal,600,'end'),phase(p,.43,.55));
 s+=rect(36,388,516,88,aligned>.95?'#eaf2ec':'#f5efe5')+txt(60,425,'Residual lag',23)+txt(527,450,`${Math.round(residual*1000)} ms`,38,aligned>.95?C.green:C.seal,600,'end');
 return s+txt(37,533,aligned>.95?'Matched events → verified resampling':'The same events arrive at different times.',24,aligned>.95?C.green:C.ink,550);
}
function controlled(p){let s=txt(36,55,'Whole-state joint error',25,C.ink,550)+txt(38,117,'Measured',23,C.cadet,550)+rect(38,135,395,34,C.cool)+rect(38,135,395*phase(p,.03,.22),34,C.cadet)+fade(txt(455,163,'0.590°',29,C.cadet,600),phase(p,.13,.22))+txt(38,234,'KineSync-GS',23,C.seal,600)+rect(38,253,395*.322/.590*phase(p,.27,.47),34,C.seal)+fade(txt(275,281,'0.322°',35,C.seal,600),phase(p,.37,.47))+txt(610,63,'Measured → verified Gaussian state',25,C.ink,550)+txt(38,361,'48 held-out conditions',24,C.ink,550);
 for(let i=0;i<48;i++)s+=rect(38+i%12*33,386+Math.floor(i/12)*31,23,21,p>.3+i*.01?(i<12?C.green:'#dce7df'):'#f0f3f1','none',3);
 return s+fade(metric(611,500,'12 / 12','Useful proposals retained',C.green,40)+metric(960,500,'0 / 48','Harmful updates',C.seal,40),phase(p,.7,.9));}
function mechanism(p){let s=txt(35,55,'Factorize calibration before recovery',25,C.ink,550)+rect(35,102,223,84,C.cool)+txt(146,137,'Shared joint',22,C.cadet,600,'middle')+txt(146,167,'calibration',22,C.cadet,600,'middle')+arrow(270,143,335,143,phase(p,.04,.22))+rect(352,87,225,56,'#f1f4f4')+txt(464,121,'Camera A residual',20,C.ink,500,'middle')+rect(352,161,225,56,'#f5efe5')+txt(464,195,'Camera B residual',20,C.ink,500,'middle')+arrow(140,209,140,255,phase(p,.18,.35))+txt(35,304,'Dual-view recovery',23,C.ink,550)+fade(txt(35,361,'42.86%',40,C.cadet,600)+arrow(202,345,272,345)+txt(299,361,'80.95%',43,C.seal,600),phase(p,.3,.52))+txt(35,420,'Successful proposals retained',23,C.ink,550);
 for(let i=0;i<18;i++)s+=circle(49+i%9*34,455+Math.floor(i/9)*34,10,p>.46+i*.02?(i<16?C.seal:'#dfe3df'):'#f0f1ef');return s+txt(370,489,'11 → 16 / 18',29,C.seal,600)+txt(665,64,'Matched calibration and update rules',25,C.ink,550)+txt(685,521,'Stronger evidence. More useful updates.',24,C.seal,550);}
function realWorld(p){return txt(35,49,'Real PiPER observations',25,C.ink,550)+txt(673,61,'Recover the corresponding Gaussian state',24,C.seal,550)+arrow(603,235,653,235,phase(p,.2,.6))+nodes(36,438,['Head + wrist images','State proposal','Cross-view verification','Synchronized state'],p,1150)+fade(txt(600,542,'Replay joint error: 84.06 → 2.26 mrad',29,C.seal,600,'middle'),phase(p,.5,.85));}
function online(p){let s=nodes(35,38,['Camera + telemetry','Temporal evidence','Spatial evidence','Verified publication'],p,1150)+txt(35,147,'Two streams, one timestamped state',26,C.ink,550);
 for(const [y,c,label] of [[218,C.gold,'Camera'],[304,C.cadet,'Telemetry']]){s+=txt(35,y-22,label,23,c,600)+line(38,y+15,552,y+15,C.line,2);for(let i=0;i<8;i++){const pos=(p*1.2+i/8)%1;s+=rect(38+pos*480,y,16,30,c,'none',3);}}
 const rows=[['Raw state','11.67','42.72','0.00'],['Temporal only','6.08','66.38','0.76'],['Spatial only','6.21','64.17','0.91'],['Unguarded fusion','3.39','84.33','6.68'],['Component update','3.82','76.48','1.37'],['KineSync-GS','3.07','83.05','0.36']];
 s+=txt(651,198,'State input',20,C.muted)+txt(911,181,'qMAE',19,C.muted,500,'middle')+txt(911,207,'mrad ↓',18,C.muted,500,'middle')+txt(1028,181,'Recovery',19,C.muted,500,'middle')+txt(1028,207,'% ↑',18,C.muted,500,'middle')+txt(1140,181,'Harmful',19,C.muted,500,'middle')+txt(1140,207,'% ↓',18,C.muted,500,'middle');
 rows.forEach((r,i)=>{const y=227+i*43;s+=rect(639,y,542,39,i===5?'#e7d6c3':i%2?'#f2f4f3':'#fafbfa');[654,911,1028,1140].forEach((x,j)=>{s+=txt(x,y+26,r[j],20,i===5?C.seal:C.ink,i===5?650:450,j?'middle':'start');});});
 return s+arrow(293,350,293,393,phase(p,.38,.65),C.green)+rect(40,416,513,75,'#eaf2ec')+txt(296,462,'qˢ(t): verified state or measurement',23,C.green,600,'middle')+fade(txt(35,542,'3.07 mrad qMAE     ·     0.36% harmful updates',27,C.seal,600),phase(p,.6,.85));}
function policy(p){return txt(36,57,'An unchanged manipulation policy',26,C.ink,550)+nodes(41,99,['State','Policy π','Action'],p,548)+txt(42,227,'Raw state',23,C.cadet,550)+rect(42,248,445*.5833*phase(p,.13,.4),36,C.cadet)+fade(txt(327,277,'58.33%',33,C.cadet,600),phase(p,.3,.4))+txt(42,345,'KineSync-GS state',23,C.seal,600)+rect(42,367,445*.7167*phase(p,.4,.68),36,C.seal)+fade(txt(387,396,'71.67%',35,C.seal,600),phase(p,.58,.68))+txt(640,68,'Real-robot task execution',25,C.ink,550)+fade(txt(600,502,'Better synchronized state improves manipulation.',31,C.seal,600,'middle')+txt(600,544,'240 paired real-robot episodes · same task and policy budget',22,C.muted,500,'middle'),phase(p,.65,.91));}
function closing(p){return txt(35,56,'An articulated state interface',27,C.seal,600)+txt(736,92,'Across simulators and policies',25,C.ink,550)+arrow(452,264,666,264,phase(p,.05,.54),C.green)+rect(474,169,189,64,'#eaf2ec')+txt(568,211,'qˢ(t)',33,C.green,600,'middle')+fade(nodes(35,472,['Propose a correction','Evaluate evidence','Synchronize the twin'],p,1150),phase(p,.25,.6));}
const SCENES={'real2sim2real':opening,failure,contract,spatial:spatialScene,temporal,controlled,mechanism,'real-world':realWorld,online,policy,closing};
export function renderScene(id,p,o={}){return `<title>${id} synchronization explanation</title>${SCENES[id](clamp(p),o)}`;}
export const sceneIds=Object.keys(SCENES);
