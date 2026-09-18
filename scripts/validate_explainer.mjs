import {readFile,access} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import path from 'node:path';
const root=path.resolve('docs');
const m=JSON.parse(await readFile(path.join(root,'static/data/explainer-timeline.json'),'utf8'));
let end=0;const hashes=new Set();
for(const s of m.scenes){
 assert.ok(Math.abs(s.start-end)<1e-5,'Non-contiguous scenes');assert.ok(s.end>s.start);end=s.end;
 assert.equal(s.cues[0].start,0);
 s.cues.forEach((c,i)=>{assert.ok(c.end>c.start);assert.ok(c.end<=s.end-s.start+1e-5);if(i)assert.equal(c.start,s.cues[i-1].end);});
 s.keyframes.forEach((k,i)=>{assert.ok(k.visual>=0&&k.visual<=1);if(i)assert.ok(k.audio>=s.keyframes[i-1].audio);});
 if(s.audio){
  const buffer=await readFile(path.join(root,s.audio.src));const hash=createHash('sha256').update(buffer).digest('hex');
  assert.equal(hash,s.audio.sha256,'Recording identity mismatch');assert.ok(!hashes.has(hash),'Duplicate active recording');hashes.add(hash);
  assert.ok(Math.abs(s.audio.duration-(s.end-s.start))<1e-5);
 }
 for(const f of ['media','poster'])if(s.evidence?.[f])await access(path.join(root,s.evidence[f]));
}
assert.ok(Math.abs(end-m.duration)<1e-5);
assert.equal(hashes.size,m.audio.recordings);
for(const f of ['explainer.html','NARRATION.md','static/data/explainer-timeline.json','static/js/tour-player.mjs','static/js/tour-scenes.mjs'])assert.ok(!/[\u4e00-\u9fff]/.test(await readFile(path.join(root,f),'utf8')),`Non-English text in ${f}`);
console.log(`Validated ${m.scenes.length} chapters, ${m.duration.toFixed(3)} seconds, ${hashes.size} unique recordings, sentence cues and media.`);
