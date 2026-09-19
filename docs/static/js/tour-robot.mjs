import * as T from 'three';
import URDFLoader from '../vendor/urdf/URDFLoader.js';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';
import {MeshSurfaceSampler} from 'three/examples/jsm/math/MeshSurfaceSampler.js';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';
import {clamp,phase,mix,spatial} from './tour-math.mjs';

const HOME=[0,-.5,0,-2.15,0,1.72,.78];
let seed=13;
const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};

// Surface samples stay in each visual's local frame. URDF forward kinematics
// transforms the Gaussian centers and tangent axes together with their link.
function surfaceGaussians(mesh){
 const sampler=new MeshSurfaceSampler(mesh).setRandomGenerator(random).build();
 const count=Math.min(1800,Math.max(90,Math.round(mesh.geometry.attributes.position.count/22)));
 const center=[],axisU=[],axisV=[],colors=[];
 const p=new T.Vector3(),n=new T.Vector3(),u=new T.Vector3(),v=new T.Vector3();
 const material=Array.isArray(mesh.material)?mesh.material[0]:mesh.material;
 const base=material.color||new T.Color('#b4bbc0');
 for(let i=0;i<count;i++){
  sampler.sample(p,n);n.normalize();
  u.crossVectors(n,Math.abs(n.z)<.9?new T.Vector3(0,0,1):new T.Vector3(0,1,0)).normalize();
  v.crossVectors(n,u).normalize();
  const a=random()*Math.PI,c=Math.cos(a),s=Math.sin(a),uu=u.clone().multiplyScalar(c).addScaledVector(v,s);
  v.multiplyScalar(c).addScaledVector(u,-s);u.copy(uu);
  const size=.004+random()*.005;
  center.push(...p.addScaledVector(n,.0008).toArray());axisU.push(...u.multiplyScalar(size*1.55).toArray());axisV.push(...v.multiplyScalar(size*.8).toArray());
  const tint=base.clone().lerp(new T.Color(i%5===0?'#b47a25':'#697f8b'),.68);
  tint.multiplyScalar(.43+.57*Math.max(0,n.dot(new T.Vector3(.4,-.65,.65).normalize())));
  colors.push(...tint.toArray());
 }
 const g=new T.InstancedBufferGeometry();
 g.setAttribute('position',new T.Float32BufferAttribute([-2,-2,0,2,-2,0,2,2,0,-2,2,0],3));g.setIndex([0,1,2,0,2,3]);
 for(const [name,array] of [['center',center],['axisU',axisU],['axisV',axisV],['tint',colors]])g.setAttribute(name,new T.InstancedBufferAttribute(new Float32Array(array),3));
 g.instanceCount=count;
 const m=new T.ShaderMaterial({transparent:true,depthWrite:false,side:T.DoubleSide,uniforms:{opacity:{value:1}},
  vertexShader:`attribute vec3 center,axisU,axisV,tint; varying vec2 uvG; varying vec3 col; void main(){uvG=position.xy;col=tint;vec3 p=center+axisU*position.x+axisV*position.y;gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.);}`,
  fragmentShader:`uniform float opacity;varying vec2 uvG;varying vec3 col;void main(){float a=exp(-1.5*dot(uvG,uvG))*opacity;if(a<.025)discard;gl_FragColor=vec4(col,a);#include <tonemapping_fragment>\n#include <colorspace_fragment>}`.replace(';#include',';\n#include')});
 const splats=new T.Mesh(g,m);splats.frustumCulled=false;splats.renderOrder=2;splats.userData.gaussians=count;return splats;
}

export class RobotStage{
 constructor(canvas,host,onChange){
  this.canvas=canvas;this.host=host;this.onChange=onChange;this.models=[];this.ready=false;this.active=false;
  this.renderer=new T.WebGLRenderer({canvas,alpha:true,antialias:true,preserveDrawingBuffer:true,powerPreference:'low-power'});
  this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));this.renderer.setClearColor(0xffffff,0);
  this.renderer.outputColorSpace=T.SRGBColorSpace;this.renderer.toneMapping=T.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.1;this.renderer.setScissorTest(true);
  this.views=Array.from({length:2},()=>this.makeView());
  this.resize=new ResizeObserver(()=>{this.size();this.paint();});this.resize.observe(host);
  this.loadPromise=this.load();
 }
 makeView(){
  const scene=new T.Scene();scene.add(new T.HemisphereLight(0xffffff,0x8b9ea5,1.0));
  const sun=new T.DirectionalLight(0xffffff,2);sun.position.set(2,-3,4);scene.add(sun);
  const fill=new T.DirectionalLight(0xe0d0b6,.6);fill.position.set(-2,1,2);scene.add(fill);
  const floor=new T.GridHelper(2.6,26,0x8b9ea5,0xcdd5d7);floor.rotation.x=Math.PI/2;floor.material.transparent=true;floor.material.opacity=.42;floor.position.z=-.005;scene.add(floor);
  const base=new T.Mesh(new T.CylinderGeometry(.13,.145,.025,40),new T.MeshStandardMaterial({color:0x8b9ea5,roughness:.65}));base.rotation.x=Math.PI/2;base.position.z=-.012;scene.add(base);
  const camera=new T.PerspectiveCamera(33,1,.01,20);camera.up.set(0,0,1);camera.position.set(1.1,-1.3,1.0);camera.lookAt(.15,0,.43);
  const target=document.createElement('div');target.className='robot-interaction';target.setAttribute('aria-label','Rotate articulated robot');target.tabIndex=0;target.title='Drag to rotate; scroll to zoom';this.host.append(target);
  const controls=new OrbitControls(camera,target);controls.target.set(.15,0,.43);controls.enablePan=false;controls.enableDamping=false;controls.minDistance=1.2;controls.maxDistance=4;controls.minPolarAngle=.15;controls.maxPolarAngle=Math.PI*.49;controls.update();
  controls.addEventListener('start',()=>this.onChange?.());controls.addEventListener('change',()=>this.paint());
  target.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight'].includes(e.key))return;e.preventDefault();e.stopPropagation();this.onChange?.();const relative=camera.position.clone().sub(controls.target);relative.applyAxisAngle(new T.Vector3(0,0,1),e.key==='ArrowLeft'?.15:-.15);camera.position.copy(relative.add(controls.target));controls.update();});
  return {scene,camera,controls,target,robot:null,meshes:[],splats:[],bounds:null};
 }
 async load(){
  const manager=new T.LoadingManager();const gltf=new GLTFLoader(manager);const loader=new URDFLoader(manager);
  loader.loadMeshCb=(url,_manager,done)=>gltf.load(url,g=>done(g.scene),undefined,e=>done(null,e));
  let loaded;const complete=new Promise((resolve,reject)=>{manager.onLoad=resolve;manager.onError=url=>reject(new Error('Robot asset unavailable: '+url));});
  loaded=await loader.loadAsync('static/models/panda/panda.urdf');await complete;
  for(const view of this.views){
   const robot=loaded.clone();view.robot=robot;view.scene.add(robot);
   const meshes=[];robot.traverse(o=>{if(o.isMesh)meshes.push(o);});
   for(const mesh of meshes){mesh.material=mesh.material.clone();view.meshes.push(mesh);const splats=surfaceGaussians(mesh);mesh.add(splats);view.splats.push(splats);}
  }
  this.ready=true;this.canvas.dataset.ready='true';this.canvas.dataset.gaussians=String(this.views[0].splats.reduce((n,s)=>n+s.userData.gaussians,0));
  if(this.last)this.update(...this.last);return this;
 }
 size(){const r=this.host.getBoundingClientRect();this.renderer.setSize(r.width,r.height,false);}
 update(id,p,options,slots){
  this.last=[id,p,options,slots];this.active=slots.length>0;this.canvas.hidden=!this.active;
  this.views.forEach((view,i)=>{
   const slot=slots[i];view.target.hidden=!slot;view.bounds=slot;if(!slot)return;
   Object.assign(view.target.style,{left:slot.x/12+'%',top:slot.y/5.6+'%',width:slot.w/12+'%',height:slot.h/5.6+'%'});
   if(!this.ready)return;
   let q=[...HOME],splat=.0;
   const motion=phase(p,.04,.92);q[0]+=.42*Math.sin(motion*Math.PI*1.3);q[1]+=.2*Math.sin(motion*Math.PI);q[3]-=.26*Math.sin(motion*Math.PI);q[5]+=.18*Math.sin(motion*Math.PI);
   if(id==='failure'){q=[...HOME];const fit=phase(p,.2,.86);q[0]+=.30*fit;q[1]-=.32*fit;q[3]+=.25*fit;splat=.85;}
   if(['spatial','contract'].includes(id)){
    const v=spatial(phase(p,.1,.62),options.branch==='conflict');const d=(id==='contract'&&!v.pass)?v.delta.map(()=>0):v.delta;
    q=HOME.map((value,j)=>value+(d[j]||0));splat=.85;
   }
   if(id==='real2sim2real')splat=i===0?0:phase(p,.24,.67);
   if(id==='closing')splat=.75;
   q.forEach((value,j)=>view.robot.setJointValue('panda_joint'+(j+1),value));view.robot.setJointValue('panda_finger_joint1',.022+.014*Math.sin(motion*Math.PI));
   if(options.representation==='mesh')splat=0;
   if(options.representation==='gaussians')splat=1;
   if(options.representation==='overlay')splat=.7;
   view.meshes.forEach(mesh=>{mesh.material.transparent=true;mesh.material.opacity=options.representation==='gaussians'?0:1-splat*.65;mesh.material.depthWrite=splat<.5;});
   view.splats.forEach(mesh=>{mesh.visible=splat>.01;mesh.material.uniforms.opacity.value=.92*splat;});
   view.robot.updateMatrixWorld(true);this.canvas.dataset.joints=q.map(v=>v.toFixed(4)).join(',');
  });this.paint();
 }
 paint(){
  if(!this.active)return;const w=this.host.clientWidth,h=this.host.clientHeight;
  this.renderer.setScissorTest(false);this.renderer.clear();this.renderer.setScissorTest(true);
  this.views.forEach(view=>{if(!view.bounds||!view.robot)return;const b=view.bounds,x=b.x/1200*w,y=h-(b.y+b.h)/560*h,width=b.w/1200*w,height=b.h/560*h;
   view.camera.aspect=width/height;view.camera.updateProjectionMatrix();this.renderer.setViewport(x,y,width,height);this.renderer.setScissor(x,y,width,height);this.renderer.render(view.scene,view.camera);
  });
 }
 dispose(){this.resize.disconnect();this.views.forEach(v=>v.controls.dispose());this.renderer.dispose();}
}
