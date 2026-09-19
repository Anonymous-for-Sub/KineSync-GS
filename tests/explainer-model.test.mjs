import test from 'node:test';
import assert from 'node:assert/strict';
import {bounded,spatial,verificationState,qMeasured,lagSearch,locate,cueAt,visualProgress} from '../docs/static/js/tour-math.mjs';
import {renderScene,sceneIds,layouts} from '../docs/static/js/tour-composition.mjs';
import * as motionMath from '../docs/static/js/tour-math.mjs';

test('opening frames isolate both robot views from recorded media',()=>{
 const svg=renderScene('real2sim2real',0,{motionTime:0});
 assert.equal((svg.match(/class="opening-world"/g)||[]).length,2);
 const media=layouts.real2sim2real.media[0];
 for(const robot of layouts.real2sim2real.robots){
  assert.ok(robot.x+robot.w<=media.x||robot.x>=media.x+media.w);
 }
 for(const id of sceneIds.filter(id=>id!=='real2sim2real')){
  assert.ok(!renderScene(id,.5,{}).includes('opening-world'));
 }
});

test('observation motion is seekable, bounded and disabled by reduced-motion preference',()=>{
 assert.equal(typeof motionMath.observationMotion,'function');
 const f=motionMath.observationMotion;
 for(const t of [0,.2,1,2.5,50]){
  const m=f(t);assert.ok(m.travel>=0&&m.travel<=1);assert.ok(m.opacity>=0&&m.opacity<=1);
  assert.deepEqual(f(t),m);assert.equal(f(t,0,true).opacity,0);
 }
 assert.notDeepEqual(f(.4),f(1.1));assert.notDeepEqual(f(.4),f(.4,.35));
});

test('flow animation changes with the narration clock but respects reduced motion',()=>{
 const a=renderScene('contract',.8,{motionTime:.4}),b=renderScene('contract',.8,{motionTime:1.1});
 assert.notEqual(a,b);
 assert.equal(renderScene('contract',.8,{motionTime:.4,reducedMotion:true}),renderScene('contract',.8,{motionTime:1.1,reducedMotion:true}));
});

test('robot and evidence regions stay inside the shared stage',()=>{
 for(const id of sceneIds)for(const r of [...layouts[id].robots,...layouts[id].media]){
  assert.ok(r.x>=0&&r.y>=0&&r.w>0&&r.h>0);
  assert.ok(r.x+r.w<=1200&&r.y+r.h<=560,`${id} exceeds stage bounds`);
 }
});

test('all proposed joint corrections respect the scalar bound',()=>{
 for(const z of [-100,-1,0,1,100])assert.ok(Math.abs(bounded([z])[0])<=.18);
});
test('publication follows evidence and never changes the conflict measurement',()=>{
 for(const p of [0,.15,.4,.55]){
  const v=verificationState(p);assert.equal(v.assessed,false);assert.equal(v.publication,0);
 }
 assert.equal(verificationState(.7).stage,'commit');assert.ok(verificationState(.7).publication>0);
 assert.equal(verificationState(1).publication,1);
 for(const p of [.1,.55,.6,.8,1])assert.equal(verificationState(p,true).publication,0);
 assert.equal(verificationState(.8,true).stage,'fallback');
 assert.deepEqual(verificationState(.4),verificationState(.4),'Seek is deterministic');
});
test('consistent views commit the complete proposal; stale evidence retains the entire measurement',()=>{
 const good=spatial(1,false),bad=spatial(1,true);
 assert.ok(good.d<.42671);assert.ok(bad.d>.42671);
 assert.deepEqual(good.output,good.proposal);assert.deepEqual(bad.output,qMeasured);
 assert.ok(good.delta.every(d=>Math.abs(d)<=.18));
});
test('correlation search recovers positive camera offsets within its sample spacing',()=>{
 for(const lag of [.1,.25,.45,.7])assert.ok(Math.abs(lagSearch(lag).best.lag-lag)<=.011);
});
test('backward seeks and exact segment boundaries resolve to the correct clock',()=>{
 const s=[{start:0,end:5},{start:5,end:9}];
 assert.deepEqual(locate(s,5),{index:1,local:0});assert.deepEqual(locate(s,9),{index:1,local:4});assert.deepEqual(locate(s,2),{index:0,local:2});
});
test('sentence cue holds and motion holds preserve recorded timing',()=>{
 const s={cues:[{start:0,end:3,text:'A'},{start:3,end:6,text:'B'}],keyframes:[{audio:0,visual:0},{audio:2,visual:.5},{audio:4,visual:.5},{audio:6,visual:1}]};
 assert.equal(cueAt(s,3).text,'B');assert.equal(visualProgress(s,3),.5);assert.equal(visualProgress(s,6),1);
});
test('all scenes reconstruct finite geometry when seeking or changing the evidence branch',()=>{
 for(const id of sceneIds)for(const p of [0,.2,.6,1])for(const branch of ['consistent','conflict']){
  const svg=renderScene(id,p,{branch,lag:.45});assert.ok(!/NaN|Infinity|undefined/.test(svg),`${id} ${p}`);assert.ok(svg.includes('<text'));
 }
});
