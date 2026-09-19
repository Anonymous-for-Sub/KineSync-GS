import * as T from 'three';
import URDFLoader from '../vendor/urdf/URDFLoader.js';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';
import {MeshSurfaceSampler} from 'three/examples/jsm/math/MeshSurfaceSampler.js';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';
import {phase,verificationState,observationMotion} from './tour-math.mjs?v=4.4';

const HOME=[0,-.5,0,-2.15,0,1.72,.78];
let seed=13;
const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};

// Surface samples stay in each visual's local frame. URDF forward kinematics
// transforms the Gaussian centers and tangent axes together with their link.
function surfaceGaussians(mesh){
 const sampler=new MeshSurfaceSampler(mesh).setRandomGenerator(random).build();
 const count=Math.min(2200,Math.max(120,Math.round(mesh.geometry.attributes.position.count/17)));
 const center=[],axisU=[],axisV=[],colors=[],accents=[];
 const p=new T.Vector3(),n=new T.Vector3(),u=new T.Vector3(),v=new T.Vector3();
 const material=Array.isArray(mesh.material)?mesh.material[0]:mesh.material;
 const base=material.color||new T.Color('#b4bbc0');
 for(let i=0;i<count;i++){
  sampler.sample(p,n);n.normalize();
  u.crossVectors(n,Math.abs(n.z)<.9?new T.Vector3(0,0,1):new T.Vector3(0,1,0)).normalize();
  v.crossVectors(n,u).normalize();
  const a=random()*Math.PI,c=Math.cos(a),s=Math.sin(a),uu=u.clone().multiplyScalar(c).addScaledVector(v,s);
  v.multiplyScalar(c).addScaledVector(u,-s);u.copy(uu);
  const accent=i%13===0?1:0;
  const size=(.0035+random()*.004)*(accent?2.3:1);
  center.push(...p.addScaledVector(n,.0008).toArray());axisU.push(...u.multiplyScalar(size*1.55).toArray());axisV.push(...v.multiplyScalar(size*.8).toArray());
  const tint=accent?new T.Color(i%2?'#cc9e4c':'#587c8c'):base.clone().lerp(new T.Color('#697f8b'),.48);
  tint.multiplyScalar(.43+.57*Math.max(0,n.dot(new T.Vector3(.4,-.65,.65).normalize())));
  colors.push(...tint.toArray());accents.push(accent);
 }
 const g=new T.InstancedBufferGeometry();
 g.setAttribute('position',new T.Float32BufferAttribute([-2,-2,0,2,-2,0,2,2,0,-2,2,0],3));g.setIndex([0,1,2,0,2,3]);
 for(const [name,array] of [['center',center],['axisU',axisU],['axisV',axisV],['tint',colors]])g.setAttribute(name,new T.InstancedBufferAttribute(new Float32Array(array),3));
 g.setAttribute('accent',new T.InstancedBufferAttribute(new Float32Array(accents),1));
 g.instanceCount=count;
 const m=new T.ShaderMaterial({transparent:true,depthWrite:false,side:T.DoubleSide,uniforms:{opacity:{value:1},scan:{value:0},scanStrength:{value:0}},
  vertexShader:`attribute vec3 center,axisU,axisV,tint;attribute float accent; varying vec2 uvG; varying vec3 col;varying float emphasis,worldZ; void main(){uvG=position.xy;col=tint;emphasis=accent;vec3 p=center+axisU*position.x+axisV*position.y;worldZ=(modelMatrix*vec4(center,1.)).z;gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.);}`,
  fragmentShader:`uniform float opacity,scan,scanStrength;varying vec2 uvG;varying vec3 col;varying float emphasis,worldZ;void main(){float r=length(uvG);float dz=worldZ-(.05+.85*scan);float band=exp(-130.*dz*dz)*scanStrength;float kernel=exp(-1.5*r*r);float contour=exp(-180.*(r-.95)*(r-.95))*emphasis*.7;float a=min(.95,(kernel*(1.-emphasis*.5)+contour)*opacity*(1.+emphasis*.4)*(1.+band*.65));if(a<.025)discard;gl_FragColor=vec4(mix(col,vec3(.7,.42,.12),band*.4),a);#include <tonemapping_fragment>\n#include <colorspace_fragment>}`.replace(';#include',';\n#include')});
 const splats=new T.Mesh(g,m);splats.frustumCulled=false;splats.renderOrder=2;splats.userData.gaussians=count;return splats;
}

// Camera bodies and view volumes illustrate the observation geometry, not calibration.
function observationCamera(color,position,lookAt){
 const group=new T.Group();group.position.set(...position);group.up.set(0,0,1);group.lookAt(new T.Vector3(...lookAt));
 const body=new T.Mesh(new T.BoxGeometry(.16,.065,.044),new T.MeshStandardMaterial({color:0x333a3d,roughness:.5}));body.position.z=-.025;group.add(body);
 const rim=new T.LineSegments(new T.EdgesGeometry(body.geometry),new T.LineBasicMaterial({color}));body.add(rim);
 for(const x of [-.053,.053]){
  const lens=new T.Mesh(new T.CylinderGeometry(.021,.024,.016,24),new T.MeshStandardMaterial({color:0x172d39,metalness:.45,roughness:.22}));lens.rotation.x=Math.PI/2;lens.position.set(x,0,.006);group.add(lens);
  const ring=new T.Mesh(new T.TorusGeometry(.023,.003,6,24),new T.MeshBasicMaterial({color}));ring.position.set(x,0,.015);group.add(ring);
 }
 const led=new T.Mesh(new T.SphereGeometry(.005,8,6),new T.MeshBasicMaterial({color}));led.position.set(0,.018,.003);group.add(led);
 const reach=group.position.distanceTo(new T.Vector3(...lookAt)),w=reach*.24,h=w*.64;
 const corners=[[-w,-h,reach],[w,-h,reach],[w,h,reach],[-w,h,reach]],lines=[];
 for(let i=0;i<4;i++)lines.push(0,0,.02,...corners[i],...corners[i],...corners[(i+1)%4]);
 const edges=new T.LineSegments(new T.BufferGeometry().setAttribute('position',new T.Float32BufferAttribute(lines,3)),new T.LineBasicMaterial({color,transparent:true,opacity:.58}));group.add(edges);
 const plane=new T.Mesh(new T.PlaneGeometry(w*2,h*2),new T.MeshBasicMaterial({color,transparent:true,opacity:.065,side:T.DoubleSide,depthWrite:false}));plane.position.z=reach;group.add(plane);
 const capture=new T.Group();
 const capturePlane=new T.Mesh(new T.PlaneGeometry(w*2,h*2),new T.MeshBasicMaterial({color,transparent:true,opacity:.12,side:T.DoubleSide,depthWrite:false}));
 const captureEdge=new T.LineSegments(new T.EdgesGeometry(capturePlane.geometry),new T.LineBasicMaterial({color,transparent:true,opacity:.8}));
 capture.add(capturePlane,captureEdge);group.add(capture);group.userData.capture={group:capture,plane:capturePlane,edge:captureEdge,reach};
 return group;
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
  const displacement=new T.Line(new T.BufferGeometry().setFromPoints([new T.Vector3(),new T.Vector3()]),new T.LineDashedMaterial({color:0xcc9e4c,dashSize:.009,gapSize:.006,transparent:true,opacity:.9,depthTest:false}));
  const markers=[0x8b9ea5,0xcc9e4c].map(color=>new T.Mesh(new T.SphereGeometry(.007,12,8),new T.MeshBasicMaterial({color,depthTest:false})));
  scene.add(displacement,...markers);displacement.visible=false;markers.forEach(m=>m.visible=false);
  const cameras=[observationCamera(0x638692,[-.32,-.27,.8],[.22,0,.55]),observationCamera(0xcc9e4c,[.62,.24,.83],[.22,0,.55])];
  scene.add(...cameras);cameras.forEach(c=>c.visible=false);
  return {scene,camera,controls,target,robot:null,meshes:[],splats:[],bounds:null,ghost:null,ghostMaterials:[],ghostEdges:[],displacement,markers,cameras};
 }
 async load(){
  const manager=new T.LoadingManager();const gltf=new GLTFLoader(manager);const loader=new URDFLoader(manager);
  loader.loadMeshCb=(url,_manager,done)=>gltf.load(url,g=>done(g.scene),undefined,e=>done(null,e));
  let loaded;const complete=new Promise((resolve,reject)=>{manager.onLoad=resolve;manager.onError=url=>reject(new Error('Robot asset unavailable: '+url));});
  loaded=await loader.loadAsync('static/models/panda/panda.urdf');await complete;
  for(const view of this.views){
   seed=13;
   const robot=loaded.clone();view.robot=robot;view.scene.add(robot);
   view.ghost=loaded.clone();view.scene.add(view.ghost);view.ghost.visible=false;
   view.ghost.traverse(o=>{if(o.isMesh){
    o.material=new T.MeshBasicMaterial({color:0xcc9e4c,transparent:true,opacity:.46,depthWrite:false,depthTest:false});o.renderOrder=4;view.ghostMaterials.push(o.material);
    const edge=new T.LineSegments(new T.EdgesGeometry(o.geometry,42),new T.LineBasicMaterial({color:0xa16a15,transparent:true,opacity:.8,depthTest:false}));edge.renderOrder=5;o.add(edge);view.ghostEdges.push(edge.material);
   }});
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
   const cameraScene=['real2sim2real','failure','spatial'].includes(id);
   view.camera.zoom=id==='real2sim2real'?.72:id==='failure'?.76:cameraScene?.86:1;
   view.cameras.forEach((c,j)=>{c.visible=cameraScene&&(id!=='real2sim2real'||j===0);});
   view.cameras.forEach((c,j)=>{
    const m=observationMotion(options.motionTime??0,j*.35,options.reducedMotion),capture=c.userData.capture;
    capture.group.visible=c.visible&&m.opacity>.01;capture.group.position.z=capture.reach*m.travel;capture.group.scale.set(m.travel,m.travel,1);
    capture.plane.material.opacity=.14*m.opacity;capture.edge.material.opacity=.75*m.opacity;
   });
   this.canvas.dataset.cameras=String(view.cameras.filter(c=>c.visible).length);
   Object.assign(view.target.style,{left:slot.x/12+'%',top:slot.y/5.6+'%',width:slot.w/12+'%',height:slot.h/5.6+'%'});
   if(!this.ready)return;
   let q=[...HOME],splat=.0,ghostQ=null,ghostOpacity=.19,ghostColor=0xcc9e4c;
   const motion=phase(p,.04,.92);q[0]+=.42*Math.sin(motion*Math.PI*1.3);q[1]+=.2*Math.sin(motion*Math.PI);q[3]-=.26*Math.sin(motion*Math.PI);q[5]+=.18*Math.sin(motion*Math.PI);
   if(id==='failure'){q=[...HOME];const fit=phase(p,.2,.86);q[0]+=.30*fit;q[1]-=.32*fit;q[3]+=.25*fit;splat=.62;ghostQ=[...HOME];ghostColor=0x8b9ea5;ghostOpacity=.24*fit;}
   if(['spatial','contract'].includes(id)){
    const v=verificationState(p,options.branch==='conflict');
    q=HOME.map((value,j)=>value+(v.delta[j]||0)*v.publication);
    ghostQ=HOME.map((value,j)=>value+(v.delta[j]||0));
    ghostOpacity=.48*phase(p,.06,.3)*(v.pass?1-phase(p,.87,.98):1);
    splat=.6;
    this.canvas.dataset.verification=v.stage;
   }
   if(id==='real2sim2real')splat=i===0?0:1;
   if(id==='closing')splat=.58;
   const gripper=['spatial','contract','failure'].includes(id)?.022:.022+.014*Math.sin(motion*Math.PI);
   q.forEach((value,j)=>view.robot.setJointValue('panda_joint'+(j+1),value));view.robot.setJointValue('panda_finger_joint1',gripper);
   this.canvas.dataset.gripper=String(gripper);
   const mode=id==='real2sim2real'&&i===0?'mesh':options.representation;
   if(mode==='mesh')splat=0;
   if(mode==='gaussians')splat=1;
   if(mode==='overlay')splat=.7;
   const pureGaussians=mode==='gaussians'||(id==='real2sim2real'&&i===1&&splat>.98);
   const buildingGaussians=id==='real2sim2real'&&i===1&&!mode;
   const meshOpacity=buildingGaussians?1-phase(splat,.5,1):pureGaussians?0:1;
   view.meshes.forEach(mesh=>{mesh.material.transparent=meshOpacity<1;mesh.material.opacity=meshOpacity;mesh.material.depthWrite=meshOpacity>.2;});
   const observation=observationMotion(options.motionTime??0,0,options.reducedMotion);
   view.splats.forEach(mesh=>{mesh.visible=splat>.01;mesh.material.uniforms.opacity.value=pureGaussians?.94:buildingGaussians?.94*splat:.4*splat;mesh.material.uniforms.scan.value=observation.scan;mesh.material.uniforms.scanStrength.value=options.reducedMotion?0:.85;});
   view.robot.updateMatrixWorld(true);
   view.ghost.visible=Boolean(ghostQ)&&ghostOpacity>.005;
   this.canvas.dataset.proposalOpacity=String(ghostQ?ghostOpacity:0);
   view.displacement.visible=view.ghost.visible;view.markers.forEach(m=>m.visible=view.ghost.visible);
   if(ghostQ){
    ghostQ.forEach((value,j)=>view.ghost.setJointValue('panda_joint'+(j+1),value));view.ghost.setJointValue('panda_finger_joint1',.022);view.ghost.updateMatrixWorld(true);
    view.ghostMaterials.forEach(m=>{m.opacity=ghostOpacity;m.color.setHex(ghostColor);});
    view.ghostEdges.forEach(m=>{m.opacity=Math.min(.9,ghostOpacity*1.8);m.color.setHex(id==='failure'?0x638692:0xa16a15);});
    const a=view.robot.links.panda_hand.getWorldPosition(new T.Vector3()),b=view.ghost.links.panda_hand.getWorldPosition(new T.Vector3());
    const positions=view.displacement.geometry.attributes.position;positions.setXYZ(0,a.x,a.y,a.z);positions.setXYZ(1,b.x,b.y,b.z);positions.needsUpdate=true;view.displacement.computeLineDistances();
    view.markers[0].position.copy(a);view.markers[1].position.copy(b);
    view.markers[0].material.color.setHex(id==='failure'?0xcc9e4c:0x8b9ea5);view.markers[1].material.color.setHex(ghostColor);
   }
   this.canvas.dataset.joints=q.map(v=>v.toFixed(4)).join(',');
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
