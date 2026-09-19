import {clamp,locate,cueAt,visualProgress} from './tour-math.mjs';
import {renderScene,layouts,evidenceFor} from './tour-composition.mjs';

const $=id=>document.getElementById(id);
const manifest=await fetch('static/data/explainer-timeline.json?v=4').then(r=>{if(!r.ok)throw new Error('Timeline unavailable');return r.json();});
const scenes=manifest.scenes;
const objectUrls=[];
const videoBuffers=new Map();
function videoURL(src){
  if(!videoBuffers.has(src))videoBuffers.set(src,fetch(src).then(r=>{if(!r.ok)throw new Error('Video unavailable');return r.blob();}).then(blob=>{const url=URL.createObjectURL(blob);objectUrls.push(url);return url;}).catch(error=>{videoBuffers.delete(src);throw error;}));
  return videoBuffers.get(src);
}
const audio=scenes.map(s=>s.audio?new Audio():null);
// Buffer short recordings once so seeking also works on hosts without byte ranges.
async function loadRecording(s,i){
  if(!s.audio)return null;
  try{
    const response=await fetch(s.audio.src);if(!response.ok)throw new Error('Recording unavailable');
    const url=URL.createObjectURL(await response.blob());objectUrls.push(url);
    await new Promise((resolve,reject)=>{audio[i].addEventListener('loadedmetadata',resolve,{once:true});audio[i].addEventListener('error',reject,{once:true});audio[i].src=url;audio[i].preload='auto';});
    return audio[i];
  }catch(error){return {error};}
}
const audioReady=scenes.map(loadRecording);
let index=-1,time=0,local=0,playing=false,muted=false,token=0,silentStart=0,silentOffset=0;
let media=[],robot=null,inspection=null,lastDraw=-1,frame=0,lastMarkup='',lastSubtitle='',loading=false;
const clock=t=>`${Math.floor(t/60).toString().padStart(2,'0')}:${Math.floor(t%60).toString().padStart(2,'0')}`;
const icon=(id,name)=>$(id).src=`static/icons/${name}.svg`;
const options=()=>inspection||{branch:'consistent',lag:.45};
function draw(force=false){
  const scene=scenes[index];if(!scene)return;
  const p=inspection?Math.max(.95,visualProgress(scene,local)):visualProgress(scene,local);
  const markup=renderScene(scene.id,p,options());
  if(markup!==lastMarkup||force){$('stage').innerHTML=markup;lastMarkup=markup;}
  robot?.update(scene.id,p,options(),layouts[scene.id].robots);
  const subtitle=cueAt(scene,local).text;
  if(subtitle!==lastSubtitle){$('subtitle').textContent=subtitle;lastSubtitle=subtitle;}
  $('time').textContent=clock(time);$('timeline').value=time;
  for(const item of media)if(item instanceof HTMLVideoElement && Number.isFinite(item.duration)){
    const target=Math.min(local,Math.max(0,item.duration-.04));
    if(Math.abs(item.currentTime-target)>.4)item.currentTime=target;
    if(playing&&local<item.duration-.05&&item.paused)item.play().catch(()=>{});
    if(!playing||local>=item.duration-.05)item.pause();
  }
}
function status(){
  icon('play-icon',playing?'pause':'play');
  $('play-label').textContent=playing?'Pause':time>=manifest.duration?'Replay tour':audio[index]?'Play with narration':'Play method interlude';
  $('play').setAttribute('aria-label',playing?'Pause presentation':'Play presentation');
}
function activate(next){
  if(next===index)return;
  if(index>=0)audio[index]?.pause();
  media.forEach(m=>m.pause?.());index=next;const scene=scenes[index];inspection=null;
  $('title').textContent=scene.title;
  $('representation-controls').hidden=!layouts[scene.id].robots.length;
  document.querySelectorAll('[data-representation]').forEach(b=>b.setAttribute('aria-pressed','false'));
  $('chapter-number').textContent=`${String(index+1).padStart(2,'0')} / 11`;
  $('branch-controls').hidden=!['contract','spatial'].includes(scene.id);$('timing-controls').hidden=scene.id!=='temporal';
  document.querySelectorAll('[data-timing]').forEach(b=>b.setAttribute('aria-pressed','false'));
  document.querySelectorAll('[data-branch]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.branch==='consistent')));
  $('lag').value=.45;$('lag-value').textContent='450 ms';
  document.querySelectorAll('[data-timing]').forEach(b=>b.setAttribute('aria-pressed','false'));
  $('stage-media').replaceChildren();media=[];
  evidenceFor(scene).forEach((evidence,i)=>{
    const b=layouts[scene.id].media[i];if(!b)return;
    const figure=document.createElement('figure');figure.className='stage-evidence';
    Object.assign(figure.style,{left:b.x/12+'%',top:b.y/5.6+'%',width:b.w/12+'%',height:b.h/5.6+'%'});
    const item=document.createElement(evidence.media.endsWith('.mp4')?'video':'img');
    if(item instanceof HTMLVideoElement){
      item.muted=true;item.playsInline=true;item.preload='auto';item.style.opacity='0';
      if(evidence.poster){item.poster=evidence.poster;figure.style.backgroundImage=`url('${evidence.poster}')`;figure.style.backgroundSize='contain';figure.style.backgroundPosition='center';figure.style.backgroundRepeat='no-repeat';}
      item.addEventListener('loadedmetadata',()=>{if(item.isConnected)draw(true);});
      item.addEventListener('seeked',()=>{item.style.opacity='1';});item.addEventListener('loadeddata',()=>{if(item.currentTime<.1)item.style.opacity='1';});
      videoURL(evidence.media).then(url=>{if(item.isConnected)item.src=url;}).catch(()=>{if(item.isConnected)item.src=evidence.media;});
    }
    else{item.src=evidence.media;item.alt=evidence.label;item.tabIndex=0;item.setAttribute('role','button');item.setAttribute('aria-label','Enlarge '+evidence.label);item.addEventListener('click',()=>openImage(evidence));item.addEventListener('keydown',e=>{if(e.key==='Enter')openImage(evidence);});}
    figure.append(item);$('stage-media').append(figure);media.push(item);
  });
  document.querySelectorAll('#chapters button').forEach((b,i)=>{if(i===index)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
  status();
}
function pause(){
  if(playing&&!loading){if(audio[index]){local=audio[index].currentTime;time=scenes[index].start+local;}else{local=clamp(silentOffset+(performance.now()-silentStart)/1000,0,scenes[index].end-scenes[index].start);time=scenes[index].start+local;}}
  playing=false;loading=false;token++;audio.forEach(a=>a?.pause());media.forEach(m=>m.pause?.());status();draw(true);
}
async function play(){
  if(time>=manifest.duration-.01){seek(0,false);}
  inspection=null;$('playback-error').hidden=true;playing=true;const ticket=++token;status();
  document.querySelectorAll('[data-representation]').forEach(b=>b.setAttribute('aria-pressed','false'));
  document.querySelectorAll('[data-branch]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.branch==='consistent')));
  $('lag').value=.45;$('lag-value').textContent='450 ms';
  const a=audio[index];
  if(a){
    const desired=local;loading=true;
    a.muted=muted;
    try{
      let ready=await audioReady[index];
      if(ready?.error&&ticket===token){audioReady[index]=loadRecording(scenes[index],index);ready=await audioReady[index];}
      if(ticket!==token)return;
      if(ready?.error)throw ready.error;
      if(Math.abs(a.currentTime-desired)>.08)a.currentTime=desired;
      await a.play();
      if(ticket!==token&&(!playing||a!==audio[index]))a.pause();
      if(ticket===token)loading=false;
    }catch(error){
      if(ticket!==token)return;
      playing=false;loading=false;status();$('playback-error').hidden=false;
      $('playback-error').textContent='Audio could not start. Press play to retry; the timeline and chapter controls remain available.';
    }
  }else{silentStart=performance.now();silentOffset=local;}
  draw(true);
}
function seek(target,resume=playing){
  token++;playing=false;loading=false;audio.forEach(a=>a?.pause());media.forEach(m=>m.pause?.());
  time=clamp(target,0,manifest.duration);const position=locate(scenes,time);activate(position.index);local=position.local;inspection=null;
  if(audio[index]){try{audio[index].currentTime=local;}catch{}}
  silentOffset=local;silentStart=performance.now();status();draw(true);
  if(resume)void play();
}
function advance(){
  if(!playing||loading)return;
  if(index===scenes.length-1){time=manifest.duration;local=scenes[index].end-scenes[index].start;playing=false;status();draw(true);return;}
  seek(scenes[index+1].start,true);
}
audio.forEach((a,i)=>{if(a){a.addEventListener('ended',()=>{if(playing&&index===i)advance();});a.addEventListener('error',()=>{if(index===i){pause();$('playback-error').hidden=false;$('playback-error').textContent='The recording could not be loaded. Press play to retry.';}});}});
function tick(now){
  frame=requestAnimationFrame(tick);
  if(!playing||loading)return;
  if(audio[index])local=audio[index].currentTime;
  else local=silentOffset+(now-silentStart)/1000;
  const duration=scenes[index].end-scenes[index].start;
  time=scenes[index].start+clamp(local,0,duration);
  if(!audio[index]&&local>=duration){advance();return;}
  if(now-lastDraw>32){draw();lastDraw=now;}
}
$('duration').textContent=clock(manifest.duration);$('timeline').max=manifest.duration;
scenes.forEach((scene,i)=>{
  const button=document.createElement('button');button.type='button';const n=document.createElement('span');n.textContent=String(i+1).padStart(2,'0');button.append(n,document.createTextNode(scene.shortTitle));
  button.addEventListener('click',()=>{seek(scene.start,false);$('chapters').hidden=true;$('chapters-toggle').setAttribute('aria-expanded','false');});$('chapters').append(button);
  const mark=document.createElement('i');mark.style.left=`${scene.start/manifest.duration*100}%`;$('timeline-marks').append(mark);
});
$('play').addEventListener('click',()=>playing?pause():void play());
$('prev').addEventListener('click',()=>seek(scenes[Math.max(0,index-1)].start));
$('next').addEventListener('click',()=>seek(scenes[Math.min(scenes.length-1,index+1)].start));
$('replay').addEventListener('click',()=>seek(scenes[index].start,true));
$('timeline').addEventListener('input',event=>seek(Number(event.target.value)));
$('mute').addEventListener('click',()=>{muted=!muted;audio.forEach(a=>{if(a)a.muted=muted;});icon('mute-icon',muted?'volume-x':'volume-2');$('mute').setAttribute('aria-pressed',String(muted));$('mute').setAttribute('aria-label',muted?'Unmute narration':'Mute narration');$('mute').title=muted?'Unmute narration':'Mute narration';});
$('fullscreen').addEventListener('click',()=>{if(document.fullscreenElement)document.exitFullscreen();else $('presentation').requestFullscreen().catch(()=>{});});
$('chapters-toggle').addEventListener('click',()=>{$('chapters').hidden=!$('chapters').hidden;$('chapters-toggle').setAttribute('aria-expanded',String(!$('chapters').hidden));});
document.querySelectorAll('[data-branch]').forEach(button=>button.addEventListener('click',()=>{pause();inspection={branch:button.dataset.branch,lag:.45};document.querySelectorAll('[data-branch]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));draw(true);}));
$('lag').addEventListener('input',()=>{pause();const lag=Number($('lag').value);inspection={branch:'consistent',lag,aligned:inspection?.aligned??0};$('lag-value').textContent=`${Math.round(lag*1000)} ms`;document.querySelectorAll('[data-timing]').forEach(b=>b.setAttribute('aria-pressed',String((b.dataset.timing==='aligned')===Boolean(inspection.aligned))));draw(true);});
document.querySelectorAll('[data-timing]').forEach(button=>button.addEventListener('click',()=>{pause();inspection={branch:'consistent',lag:Number($('lag').value),aligned:button.dataset.timing==='aligned'?1:0};document.querySelectorAll('[data-timing]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));draw(true);}));
document.querySelectorAll('[data-representation]').forEach(button=>button.addEventListener('click',()=>{pause();inspection={...options(),representation:button.dataset.representation};document.querySelectorAll('[data-representation]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));draw(true);}));
document.addEventListener('keydown',event=>{if(/INPUT|BUTTON|SELECT|TEXTAREA/.test(event.target.tagName))return;if(event.code==='Space'){event.preventDefault();playing?pause():void play();}if(event.code==='ArrowRight')seek(time+5);if(event.code==='ArrowLeft')seek(time-5);if(event.code==='Escape'){$('chapters').hidden=true;$('chapters-toggle').setAttribute('aria-expanded','false');}});
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
window.addEventListener('pagehide',()=>{pause();cancelAnimationFrame(frame);robot?.dispose();objectUrls.forEach(url=>URL.revokeObjectURL(url));});
const requested=Number(new URLSearchParams(location.search).get('t')||0);seek(Number.isFinite(requested)?requested:0,false);frame=requestAnimationFrame(tick);
window.kineSyncTour={seek,pause,play,get state(){return {index,time,local,playing,muted,audioTime:audio[index]?.currentTime??null,duration:manifest.duration};}};

function openImage(evidence){
  pause();const dialog=document.createElement('dialog');dialog.className='media-dialog';
  const img=document.createElement('img');img.src=evidence.media;img.alt=evidence.label;
  const close=document.createElement('button');close.textContent='×';close.setAttribute('aria-label','Close image');close.onclick=()=>dialog.close();
  dialog.append(close,img);document.body.append(dialog);dialog.addEventListener('close',()=>dialog.remove());dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});dialog.showModal();
}
try{
  const {RobotStage}=await import('./tour-robot.mjs');
  robot=new RobotStage($('robot-canvas'),$('scene-surface'),pause);draw(true);
  await robot.loadPromise;draw(true);
}catch(error){
  $('robot-error').hidden=false;$('robot-error').textContent='The 3D view could not load. Reload to retry; narration and experiment media remain available.';
  console.error(error);
}
