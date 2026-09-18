import test from 'node:test';
import assert from 'node:assert/strict';
import {bounded,spatial,qMeasured,lagSearch,locate,cueAt,visualProgress} from '../docs/static/js/tour-math.mjs';
import {renderScene,sceneIds} from '../docs/static/js/tour-scenes.mjs';

test('all proposed joint corrections respect the scalar bound',()=>{
 for(const z of [-100,-1,0,1,100])assert.ok(Math.abs(bounded([z])[0])<=.18);
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
