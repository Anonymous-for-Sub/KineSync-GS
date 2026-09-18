import {C,clamp,mix,ease,phase,forward,qMeasured,spatial,signal,lagSearch,RESULTS as R} from './tour-math.mjs';

const text=(x,y,s,size=18,color=C.ink,weight=400,anchor='start')=>`<text x="${x}" y="${y}" fill="${color}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${s}</text>`;
const rect=(x,y,w,h,fill='none',stroke=C.line,r=5)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}"/>`;
const line=(x,y,u,v,c=C.line,w=1,dash='')=>`<path d="M${x},${y} L${u},${v}" stroke="${c}" stroke-width="${w}" fill="none" ${dash?`stroke-dasharray="${dash}"`:''}/>`;
const group=(body,p=1)=>`<g opacity="${clamp(p)}">${body}</g>`;
const circle=(x,y,r,c,stroke='none')=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${c}" stroke="${stroke}"/>`;
const arrow=(x,y,u,v,p=1,c=C.gold)=>{const xx=mix(x,u,p),yy=mix(y,v,p),a=Math.atan2(v-y,u-x);return line(x,y,u,v,C.line,2)+line(x,y,xx,yy,c,2)+`<path d="M-7,-5 L0,0 L-7,5" transform="translate(${xx},${yy}) rotate(${a*180/Math.PI})" fill="none" stroke="${c}" stroke-width="2"/>`;};
const rule=(x,y,w)=>line(x,y,x+w,y);
function grid(x,y,w,h,spacing=30){let s='';for(let xx=x;xx<=x+w;xx+=spacing)s+=line(xx,y,xx,y+h,'#f0f2ef');for(let yy=y;yy<=y+h;yy+=spacing)s+=line(x,yy,x+w,yy,'#f0f2ef');return s;}
function robot(q,x,y,scale=1,color=C.cadet,splats=false,ghost=false){
 const pts=forward(q,[0,0]);let b=rect(-27,-5,54,20,ghost?'none':'#eff1ed',ghost?color:'#b9c1bd',3);
 for(let i=0;i<pts.length-1;i++){
  const [a,z]=[pts[i],pts[i+1]];const width=i<3?22:13;
  b+=`<path d="M${a} L${z}" fill="none" stroke="${ghost?color:'#444d49'}" stroke-width="${ghost?2:width+2}" stroke-linecap="round" ${ghost?'stroke-dasharray="5 5"':''}/>`;
  if(!ghost)b+=`<path d="M${a} L${z}" fill="none" stroke="${splats?'#faf7f0':'#f0f3f1'}" stroke-width="${width}" stroke-linecap="round"/>`;
  if(splats&&!ghost)for(let j=0;j<9;j++){const t=(j+.5)/9;const angle=Math.atan2(z[1]-a[1],z[0]-a[0]);const xx=mix(a[0],z[0],t),yy=mix(a[1],z[1],t);b+=`<ellipse cx="${xx}" cy="${yy}" rx="10" ry="${4+(j%3)}" transform="rotate(${angle*180/Math.PI} ${xx} ${yy})" fill="${color}" opacity="${.25+(j%4)*.12}"/>`;}
  b+=circle(...a,ghost?4:width*.58,ghost?'#fff':color,ghost?color:'#fff');
 }
 const tip=pts.at(-1);b+=`<path d="M${tip[0]-8},${tip[1]} v17 h5 M${tip[0]+8},${tip[1]} v17 h-5" fill="none" stroke="${color}" stroke-width="4"/>`;
 return `<g transform="translate(${x},${y}) scale(${scale})" opacity="${ghost?.55:1}">${b}</g>`;
}
function camera(x,y,label,p=1){return group(rect(x,y,66,43,'#fff',C.cadet)+circle(x+33,y+21,13,'#fff',C.cadet)+circle(x+33,y+21,6,C.gold)+line(x+21,y+44,x+15,y+56,C.cadet,2)+line(x+45,y+44,x+51,y+56,C.cadet,2)+text(x+33,y-13,label,14,C.muted,500,'middle'),p);}
function badge(x,y,label,pass=true){const color=pass?C.green:C.seal;return rect(x,y,180,38,pass?'#eef5ef':'#f7efeb','none')+circle(x+18,y+19,4,color)+text(x+32,y+25,label,15,color,600);}
function stateVector(x,y,v,color=C.gold,label='CORRECTION',p=1){let s=text(x,y-24,label,13,C.muted,600);v.forEach((a,i)=>{s+=rect(x+i*36,y,28,80,'#f4f5f2','none',3)+line(x+i*36,y+40,x+i*36+28,y+40,'#cbd1cb')+rect(x+i*36+5,a>=0?y+40-a*190*p:y+40,18,Math.abs(a)*190*p,color,'none',1)+text(x+i*36+14,y+101,`q${i+1}`,12,C.muted,400,'middle');});return s;}
const note=(x,y,a,b)=>text(x,y,a,14,C.muted,500)+text(x,y+28,b,23,C.ink,600);
function particles(x,y,u,v,p,color=C.gold){let s='';for(let i=0;i<7;i++){const t=clamp(p*1.4-i*.06);s+=circle(mix(x,u,t),mix(y,v,t),3+i%3,color);}return s;}
const path=(points,c,width=3)=>`<path d="${points.map(([x,y],i)=>`${i?'L':'M'}${x.toFixed(2)},${y.toFixed(2)}`).join(' ')}" fill="none" stroke="${c}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round"/>`;

function opening(p){
 const move=phase(p,.12,.8);const q=qMeasured.map((v,i)=>v+Math.sin(i+1)*move*.24);
 return text(35,48,'REAL ROBOT',15,C.muted,650)+text(650,48,'GAUSSIAN TWIN',15,C.muted,650)+
 grid(30,125,370,290)+grid(605,125,370,290)+robot(q,160,407,1,C.cadet)+robot(q,735,407,1,C.gold,true)+
 rect(274,363,44,35,C.seal,'none',3)+rect(849,363,44,35,C.seal,'none',3)+
 group(line(50,130+move*250,405,130+move*250,C.gold,2),.65)+
 text(483,151,'OBSERVATION',12,C.muted,600,'middle')+rect(434,173,101,99,'#fff',C.line)+
 Array.from({length:6},(_,i)=>rect(445+i*13,190,8,Math.max(6,55*Math.abs(Math.sin(i+move*3))),i%2?C.gold:C.cadet,'none',1)).join('')+
 text(484,256,'q(t), I(t)',15,C.ink,500,'middle')+arrow(394,298,610,298,phase(p,.25,.65))+particles(405,298,602,298,move)+
 text(195,454,'Appearance',19,C.ink,500,'middle')+text(495,454,'Articulation',19,C.ink,500,'middle')+text(805,454,'Time',19,C.ink,500,'middle')+
 line(275,449,410,449)+line(570,449,745,449)+group(text(500,507,'The same robot. The same instant. The same state.',22,C.seal,550,'middle'),phase(p,.63,.93));
}
function failure(p){
 const t=phase(p,.12,.85),q=qMeasured.map((x,i)=>x+[.32,-.45,.42,-.18,.1,.1][i]*t);
 let s=camera(65,95,'Current view')+camera(297,95,'Stale view')+line(98,151,213,245,C.cadet,1,'5 5')+line(331,151,270,245,C.gold,1,'5 5');
 s+=grid(25,200,440,237)+robot(qMeasured,158,421,.91,C.cadet,false,true)+robot(q,158,421,.91,C.gold,true)+
 text(110,474,'Physical state',15,C.cadet,500)+text(285,474,'Image-fitted state',15,C.gold,500)+
 text(535,70,'A CONFLICTING OBSERVATION',13,C.muted,600)+rule(535,89,425);
 const xx=575,ww=320;const vals=Array.from({length:60},(_,i)=>i/59*t);
 s+=text(535,131,'Image residual',17,C.ink,500)+text(930,131,'↓',22,C.green,650)+grid(xx,151,ww,98,32)+path(vals.map(u=>[xx+ww*u,155+82*(1-Math.exp(-u*3))]),C.gold,4)+circle(xx+ww*t,155+82*(1-Math.exp(-t*3)),5,C.gold);
 s+=text(535,301,'Joint-state error',17,C.ink,500)+text(930,301,'↑',22,C.seal,650)+grid(xx,324,ww,100,32)+path(vals.map(u=>[xx+ww*u,419-88*u*u]),C.seal,4)+circle(xx+ww*t,419-88*t*t,5,C.seal)+text(737,458,'Fitting iterations',13,C.muted,400,'middle');
 return s+group(text(532,506,'Image agreement ≠ state correctness',22,C.seal,550),phase(p,.68,.92));
}
function contract(p,options){
 const conflict=options.branch==='conflict';const t=phase(p,.1,.4),v=spatial(t,conflict),gate=phase(p,.4,.65),commit=phase(p,.65,.94);
 let s=text(35,48,'01  PROPOSE',15,C.gold,650)+text(379,48,'02  VERIFY',15,C.seal,650)+text(753,48,'03  PUBLISH',15,C.green,650)+rule(35,68,260)+rule(379,68,270)+rule(753,68,210);
 s+=stateVector(52,171,v.delta,C.gold,'ONE COORDINATED VECTOR')+text(52,315,'δ = b tanh(z)',24,C.ink,500)+text(52,347,'Every joint stays inside the bound.',15,C.muted)+arrow(280,210,394,210,phase(p,.25,.5));
 s+=`<path d="M515,114 L616,210 L515,306 L414,210 Z" fill="${gate>.5?(v.pass?'#eef5ef':'#f7efeb'):'#f6f6f3'}" stroke="${gate>.5?(v.pass?C.green:C.seal):C.line}" stroke-width="2"/>`+text(515,200,'Cross-view',18,C.ink,600,'middle')+text(515,229,'evidence',18,C.ink,600,'middle');
 s+=line(614,210,720,210,C.line,2)+arrow(649,210,729,150,commit,v.pass?C.green:C.line)+arrow(649,210,729,317,commit,v.pass?C.line:C.cadet);
 s+=group(rect(737,112,226,73,v.pass?'#eef5ef':'#fafbf9','none')+text(755,141,'Verified proposal',19,v.pass?C.green:C.muted,600)+text(755,167,'qˢ = q⁺',21,v.pass?C.green:C.muted),.3+.7*(v.pass?commit:0));
 s+=group(rect(737,279,226,73,!v.pass?'#edf2f3':'#fafbf9','none')+text(755,308,'Measurement retained',17,!v.pass?C.ink:C.muted,550)+text(755,335,'qˢ = qᵐ',21,!v.pass?C.cadet:C.muted),.3+.7*(!v.pass?commit:0));
 s+=line(145,376,145,431,C.cadet,2)+line(145,431,850,431,C.cadet,2,'5 5')+line(850,431,850,354,C.cadet,2,'5 5')+text(442,459,'The measured state remains available.',15,C.muted,400,'middle');
 return s+group(text(500,510,'Estimation proposes. Evidence authorizes.',25,C.seal,600,'middle'),phase(p,.63,.9));
}
function spatialScene(p,options){
 const v=spatial(phase(p,.12,.58),options.branch==='conflict');const show=phase(p,.58,.85);
 let s=grid(30,138,412,303)+camera(66,83,'View A')+camera(330,83,'View B')+line(99,142,254,271,C.cadet,1,'6 5')+line(363,142,254,271,C.gold,1,'6 5');
 s+=robot(qMeasured,155,425,.95,C.cadet,false,true)+robot(v.proposal,155,425,.95,C.gold,true)+text(53,477,'Kinematics binds every Gaussian to a link.',15,C.muted);
 s+=text(500,67,'SEPARATELY RECOVERED CORRECTIONS',14,C.muted,600)+stateVector(500,125,v.a,C.cadet,'δA · VIEW A')+stateVector(754,125,v.b,C.gold,'δB · VIEW B');
 s+=rule(500,264,463)+text(500,300,'Normalized disagreement',18,C.ink,500)+text(500,334,'d = ‖δA − δB‖ / (‖δA‖ + ‖δB‖)',20,C.ink)+
 rect(500,365,462,12,'#edf0ec','none',2)+rect(500,365,462*v.d,12,v.pass?C.green:C.seal,'none',2)+line(500+462*.42671,352,500+462*.42671,390,C.ink,2)+text(500+462*.42671,415,'τd = 0.42671',14,C.muted,400,'middle')+text(956,349,v.d.toFixed(3),22,v.pass?C.green:C.seal,650,'end');
 s+=group(badge(501,445,v.pass?'Commit whole state':'Retain measurement',v.pass),show)+group(text(708,470,v.pass?'qˢ ← q⁺':'qˢ ← qᵐ',24,v.pass?C.green:C.seal,600),show);
 return s;
}
function temporal(p,options){
 const shift=options.lag??.45,search=lagSearch(shift),aligned=options.aligned??phase(p,.52,.85),offset=shift*(1-aligned),sweep=phase(p,.18,.53);const xx=57,ww=550;
 let s=text(42,48,'CROSS-MODAL MOTION',14,C.muted,650)+text(42,93,'Telemetry',17,C.cadet,550)+text(42,232,'Image motion',17,C.gold,550);
 for(const [yy,delay,color] of [[115,0,C.cadet],[253,offset,C.gold]]){s+=grid(xx,yy,ww,86,55);const pts=Array.from({length:130},(_,i)=>{const t=i/129*4;return [xx+t/4*ww,yy+82-signal(t-delay)*77];});s+=path(pts,color,3.5);for(let i=0;i<24;i++){const t=i/6;s+=circle(xx+t/4*ww,yy+82-signal(t-delay)*77,3,color);}}
 s+=line(xx+4.2/4*ww*.3,108,xx+4.2/4*ww*.3,345,C.line,1,'3 4')+text(600,371,`Residual lag  ${(offset*1000).toFixed(0)} ms`,18,aligned>.9?C.green:C.seal,550,'end');
 s+=text(670,94,'LAG SEARCH',14,C.muted,600)+grid(675,131,276,169,46)+path(search.values.map(v=>[675+(v.lag+.8)/1.6*276,293-(v.score+1)*74]),C.seal,3);
 const scan=-.8+sweep*1.6,best=search.best;const marker=p<.54?scan:best.lag;
 s+=line(675+(marker+.8)/1.6*276,129,675+(marker+.8)/1.6*276,303,C.gold,2)+text(675,328,'−800',12,C.muted)+text(951,328,'800 ms',12,C.muted,400,'end')+group(text(812,378,`Δt⁺ = ${(best.lag*1000).toFixed(0)} ms`,23,C.seal,600,'middle'),phase(p,.46,.57));
 s+=arrow(168,415,421,415,phase(p,.55,.83),C.green)+text(47,451,'Original association',16,C.muted)+text(334,451,'Verified resampling',16,C.green,550)+group(badge(718,431,'Motion supported'),phase(p,.55,.84));
 return s+text(48,509,'The clock changes only after timing evidence supports the correction.',20,C.ink,500);
}
function controlled(p){
 let s=text(44,54,'WHOLE-STATE JOINT ERROR',15,C.muted,650)+text(44,106,'Measured state',21,C.cadet,550)+rect(44,129,407,31,'#edf2f3','none',2)+group(rect(44,129,407,31,C.cadet,'none',2),phase(p,.06,.22))+text(474,153,'0.590°',27,C.cadet,600);
 s+=text(44,220,'KineSync-GS',21,C.seal,600)+group(rect(44,243,407*.322/.590,31,C.seal,'none',2),phase(p,.25,.43))+text(288,268,'0.322°',30,C.seal,650);
 s+=group(text(44,359,'45.4%',58,C.seal,600)+text(44,395,'lower joint error',19,C.muted),phase(p,.4,.65));
 s+=text(655,54,'48 HELD-OUT CONDITIONS',15,C.muted,650)+text(655,84,'12 poses × 4 view / offset conditions',15,C.muted);
 for(let i=0;i<48;i++){const on=p>.15+i/48*.56;s+=rect(656+(i%8)*37,118+Math.floor(i/8)*38,25,25,on?(i<12?C.green:'#e6ece7'):'#f6f7f4','none',4);}
 s+=group(text(656,395,'12 / 12',37,C.green,600)+text(656,423,'useful proposals retained',16,C.muted)+text(656,478,'0 / 48 harmful updates',23,C.seal,600),phase(p,.7,.9));
 return s+text(44,504,'Selective updates preserve the useful corrections.',21,C.ink,500);
}
function mechanism(p){
 let s=text(38,49,'FACTORIZE PERSISTENT CALIBRATION',15,C.muted,650)+text(40,94,'Shared joint zero',21,C.ink,550)+text(380,94,'Camera-specific residuals',21,C.ink,550);
 for(let i=0;i<6;i++)s+=rect(43+i*40,126,27,55,C.cadet,'none',2)+text(57+i*40,208,`${i+1}`,13,C.muted,400,'middle');
 s+=text(312,163,'+',36,C.gold,500)+rect(390,125,182,75,C.cool,'none')+text(481,171,'View A',21,C.cadet,550,'middle')+rect(610,125,182,75,C.pale,'none')+text(701,171,'View B',21,C.gold,550,'middle');
 s+=arrow(230,234,445,289,phase(p,.13,.35))+arrow(705,234,553,289,phase(p,.13,.35))+group(text(500,327,'Sharper cross-view evidence',26,C.seal,600,'middle'),phase(p,.2,.4));
 s+=rule(40,351,920)+text(40,389,'DUAL-VIEW RECOVERY',13,C.muted,650)+text(550,389,'SUCCESSFUL PROPOSALS RETAINED',13,C.muted,650);
 s+=group(text(40,452,'42.86%',34,C.cadet,600)+arrow(212,438,288,438,1)+text(307,452,'80.95%',36,C.seal,600),phase(p,.35,.58));
 for(let i=0;i<18;i++){s+=circle(558+(i%9)*37,427+Math.floor(i/9)*34,10,p>.5+i*.016?(i<16?C.seal:'#e1e4df'):'#f0f2ee');}
 return s+text(40,500,'Joint calibration + view calibration',16,C.muted)+group(text(920,498,'11 → 16 of 18',22,C.seal,600,'end'),phase(p,.7,.9));
}
function realWorld(p){
 const selected=phase(p,.15,.55),q=qMeasured.map((x,i)=>x+Math.sin(i+1)*selected*.12);
 let s=camera(61,93,'Head camera')+camera(61,278,'Wrist camera')+arrow(138,120,292,166,phase(p,.08,.28))+arrow(138,305,292,231,phase(p,.08,.28));
 s+=text(300,60,'GAUSSIAN OBSERVATION MODEL',14,C.muted,650);
 for(let i=0;i<5;i++){const pos=300+i*88;const chosen=i===3&&p>.42;s+=rect(pos,117,73,136,chosen?'#f5efe5':'#f8faf7',chosen?C.gold:C.line)+robot(qMeasured.map((x,j)=>x+Math.sin(j+i)*.16),pos+27,234,.26,chosen?C.gold:C.cadet,true)+text(pos+36,276,`t${i-2>=0?'+':''}${i-2}`,14,C.muted,400,'middle');}
 s+=line(608,283,608,333,C.gold,2)+group(badge(477,350,'Cross-view support'),phase(p,.43,.68))+arrow(704,390,799,390,phase(p,.64,.86),C.green)+robot(q,842,372,.49,C.green,true);
 s+=text(76,411,'qᵐ',26,C.cadet,650)+line(113,403,394,403,C.cadet,2,'5 5')+text(859,443,'qˢ(t)',24,C.green,600,'middle');
 return s+group(text(44,504,'Different visual backends. One synchronized state.',23,C.seal,550),phase(p,.5,.86));
}
function online(p){
 let s=text(45,50,'ASYNCHRONOUS INPUTS',14,C.muted,650)+text(655,50,'VERIFIED STATE STREAM',14,C.muted,650);
 const lanes=[['Camera',112,C.gold],['Telemetry',223,C.cadet]];
 for(const [label,y,color] of lanes){s+=text(45,y-20,label,17,color,550)+line(43,y+20,325,y+20,C.line,2);for(let i=0;i<6;i++){const t=clamp(p*1.8-i*.09);const x=45+t*265;s+=rect(x,y+3,24,34,color,'none',3);}}
 s+=rect(340,102,194,173,'#f7f8f5',C.line)+text(437,144,'Temporal',23,C.ink,550,'middle')+text(437,175,'verification',19,C.ink,500,'middle')+group(text(437,227,'Verified association',14,C.green,600,'middle'),phase(p,.15,.4));
 s+=arrow(535,189,588,189,phase(p,.25,.52))+rect(604,102,190,173,'#f7f8f5',C.line)+text(699,144,'Spatial',23,C.ink,550,'middle')+text(699,175,'verification',19,C.ink,500,'middle')+group(text(699,227,'Whole-state decision',14,C.green,600,'middle'),phase(p,.45,.7));
 s+=arrow(795,189,931,189,phase(p,.55,.83),C.green)+group(circle(946,189,15,C.green)+text(931,246,'qˢ(t)',24,C.green,550,'middle'),phase(p,.68,.88));
 s+=line(122,308,122,355,C.cadet,2)+line(122,355,917,355,C.cadet,2,'5 5')+line(917,355,917,221,C.cadet,2,'5 5')+text(515,384,'Measurement-preserving fallback',16,C.muted,500,'middle');
 s+=group(note(46,453,'ONLINE qMAE','3.07 mrad')+note(395,453,'HARMFUL UPDATES','0.36%')+note(741,453,'RECOVERY','83.05%'),phase(p,.6,.87));return s;
}
function policy(p){
 let s=text(42,48,'SAME POLICY · SAME TASKS · SAME EPISODE BUDGET',15,C.muted,650);
 s+=text(58,108,'Raw state',20,C.cadet,550)+text(58,270,'Synchronized state',20,C.seal,600)+stateVector(59,129,[.08,-.1,.04,.07,-.1,.13],C.cadet,'',1)+stateVector(59,294,[.06,-.03,.09,.03,-.06,.08],C.seal,'',1);
 s+=arrow(290,171,392,225,phase(p,.06,.34),C.cadet)+arrow(290,336,392,272,phase(p,.06,.34),C.seal)+rect(412,191,178,123,'#f7f8f5',C.line)+text(501,232,'π',40,C.ink,600,'middle')+text(501,278,'UNCHANGED',12,C.muted,650,'middle')+arrow(607,252,676,252,phase(p,.25,.48));
 s+=text(687,123,'TASK-MACRO SUCCESS',14,C.muted,650)+text(687,179,'58.33%',33,C.cadet,600)+group(rect(687,195,270*.5833,27,C.cadet,'none',2),phase(p,.35,.57))+text(687,285,'71.67%',40,C.seal,600)+group(rect(687,304,270*.7167,27,C.seal,'none',2),phase(p,.54,.78));
 return s+group(rule(44,429,913)+text(44,477,'Better synchronized state → better manipulation',26,C.seal,550)+text(945,512,'240 paired real-robot episodes',16,C.muted,400,'end'),phase(p,.68,.94));
}
function closing(p){
 let s=robot(qMeasured,137,355,.65,C.cadet)+text(141,398,'Real observations',19,C.ink,550,'middle')+arrow(270,260,377,260,phase(p,.06,.36));
 s+=`<path d="M485,147 L596,260 L485,373 L374,260 Z" fill="#f5efe5" stroke="${C.gold}" stroke-width="2"/>`+text(485,233,'KineSync-GS',23,C.seal,650,'middle')+text(485,270,'Verified state',17,C.ink,500,'middle')+text(485,302,'qˢ(t)',24,C.seal,600,'middle');
 s+=arrow(599,260,748,131,phase(p,.25,.55),C.green)+arrow(599,260,748,260,phase(p,.35,.65),C.green)+arrow(599,260,748,389,phase(p,.45,.75),C.green);
 s+=group(text(766,134,'Gaussian twin',23,C.ink,550)+text(766,161,'Articulated appearance',14,C.muted),phase(p,.25,.55))+group(text(766,263,'Simulation',23,C.ink,550)+text(766,290,'A shared state interface',14,C.muted),phase(p,.35,.65))+group(text(766,392,'Robot policy',23,C.ink,550)+text(766,419,'Synchronized observations',14,C.muted),phase(p,.45,.75));
 return s+text(46,57,'FROM OBSERVATION TO STATE-CHANGING AUTHORITY',15,C.muted,650)+group(text(500,506,'Propose. Verify. Synchronize.',32,C.seal,600,'middle'),phase(p,.6,.88));
}
const SCENES={'real2sim2real':opening,failure,contract,spatial:spatialScene,temporal,controlled,mechanism,'real-world':realWorld,online,policy,closing};
export function renderScene(id,progress,options={}){return `<title>${id} synchronization explanation</title>${SCENES[id](clamp(progress),options)}`;}
export const sceneIds=Object.keys(SCENES);
