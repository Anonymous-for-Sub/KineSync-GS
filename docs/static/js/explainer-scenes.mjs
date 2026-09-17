import {
  COLORS, POSES, armPoints, clamp, ease, evidenceScores, gaussianPoints,
  interpolatePose, lerp, motionSignal, pathFromPoints, phase, sampleCurve,
  sceneState,
} from "./explainer-model.mjs";

const fmt = (value) => Number(value).toFixed(1);
const opacity = (value) => clamp(value).toFixed(3);

function defs() {
  return `<defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="${COLORS.gold}"/></marker>
    <marker id="arrow-seal" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="${COLORS.seal}"/></marker>
    <filter id="soft-shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#442c1b" flood-opacity=".13"/></filter>
  </defs>`;
}

function backdrop() {
  const vertical = Array.from({ length: 21 }, (_, i) => `<line x1="${i * 48}" y1="0" x2="${i * 48}" y2="520"/>`).join("");
  const horizontal = Array.from({ length: 12 }, (_, i) => `<line x1="0" y1="${i * 48}" x2="960" y2="${i * 48}"/>`).join("");
  return `${defs()}<rect width="960" height="520" fill="#fbfaf8"/><g stroke="#e9e2db" stroke-width="1" opacity=".58">${vertical}${horizontal}</g><path d="M0 456H960" stroke="#d9cec4" stroke-width="2"/>`;
}

function roundedCard(x, y, width, height, title, body = "", accent = COLORS.gold, active = true) {
  return `<g opacity="${active ? 1 : .42}" filter="url(#soft-shadow)">
    <rect x="${x}" y="${y}" width="${width}" height="${height}" rx="8" fill="#fff" stroke="${active ? accent : COLORS.line}" stroke-width="${active ? 2 : 1}"/>
    <rect x="${x}" y="${y}" width="5" height="${height}" rx="2.5" fill="${accent}"/>
    <text x="${x + 18}" y="${y + 27}" class="svg-title">${title}</text>
    ${body ? `<text x="${x + 18}" y="${y + 49}" class="svg-small">${body}</text>` : ""}
  </g>`;
}

function camera(x, y, label, pulse = 0) {
  return `<g transform="translate(${x} ${y})">
    <rect x="0" y="0" width="62" height="42" rx="7" fill="#fff" stroke="${COLORS.cadet}" stroke-width="2"/>
    <circle cx="31" cy="21" r="11" fill="${COLORS.soft}" stroke="${COLORS.seal}" stroke-width="2"/>
    <circle cx="31" cy="21" r="${4 + pulse * 2}" fill="${COLORS.gold}" opacity="${.6 + pulse * .4}"/>
    <path d="M14 42L7 56M48 42L55 56" stroke="${COLORS.cadet}" stroke-width="3" stroke-linecap="round"/>
    <text x="31" y="72" text-anchor="middle" class="svg-small">${label}</text>
  </g>`;
}

function robot(pose, options = {}) {
  const {
    base = [180, 430], color = COLORS.measurement, opacity: alpha = 1,
    dashed = false, splats = false, label = "", width = 17, gripper = true,
  } = options;
  const points = armPoints(pose, base);
  const links = points.slice(0, -1).map(([x, y], index) => {
    const [nx, ny] = points[index + 1];
    return `<line x1="${fmt(x)}" y1="${fmt(y)}" x2="${fmt(nx)}" y2="${fmt(ny)}" stroke="${color}" stroke-width="${width - index * 1.4}" stroke-linecap="round" ${dashed ? 'stroke-dasharray="8 7"' : ""}/>`;
  }).join("");
  const joints = points.map(([x, y], index) => `<circle cx="${fmt(x)}" cy="${fmt(y)}" r="${index ? 8 : 12}" fill="#fff" stroke="${color}" stroke-width="4"/>`).join("");
  const blobs = splats ? gaussianPoints(points, 6).map(([x, y, r], index) => `<ellipse cx="${fmt(x)}" cy="${fmt(y)}" rx="${r + 2}" ry="${r}" fill="${index % 3 ? COLORS.gold : color}" opacity=".38" transform="rotate(${(index * 29) % 180} ${fmt(x)} ${fmt(y)})"/>`).join("") : "";
  const tip = points[points.length - 1];
  const claw = gripper ? `<path d="M${fmt(tip[0])} ${fmt(tip[1])}l-13 -10M${fmt(tip[0])} ${fmt(tip[1])}l12 -11" stroke="${color}" stroke-width="5" stroke-linecap="round"/>` : "";
  return `<g opacity="${alpha}">${blobs}${links}${joints}${claw}${label ? `<text x="${base[0]}" y="${base[1] + 32}" text-anchor="middle" class="svg-label">${label}</text>` : ""}</g>`;
}

function arrow(x1, y1, x2, y2, progress = 1, color = COLORS.gold, dashed = false) {
  const x = lerp(x1, x2, clamp(progress));
  const y = lerp(y1, y2, clamp(progress));
  return `<line x1="${x1}" y1="${y1}" x2="${fmt(x)}" y2="${fmt(y)}" stroke="${color}" stroke-width="3" ${dashed ? 'stroke-dasharray="7 6"' : ""} marker-end="url(#arrow)"/>`;
}

function gauge(x, y, width, label, value, color = COLORS.gold) {
  return `<g><text x="${x}" y="${y - 7}" class="svg-small">${label}</text><rect x="${x}" y="${y}" width="${width}" height="8" rx="4" fill="#ece6e0"/><rect x="${x}" y="${y}" width="${width * clamp(value)}" height="8" rx="4" fill="${color}"/><text x="${x + width + 8}" y="${y + 8}" class="svg-small">${Math.round(value * 100)}%</text></g>`;
}

function sceneReal2Sim(progress) {
  const move = Math.sin(progress * Math.PI * 1.4) * .18;
  const pose = [POSES.measuredPose[0] + move, ...POSES.measuredPose.slice(1)];
  const frameProgress = phase(progress, .12, .72);
  const packetX = lerp(330, 640, frameProgress);
  return `${backdrop()}
    <text x="64" y="46" class="svg-label">PHYSICAL ROBOT</text>
    <text x="724" y="46" class="svg-label">ARTICULATED GAUSSIAN TWIN</text>
    ${camera(282, 68, "RGB-D camera", .5 + .5 * Math.sin(progress * 18))}
    ${robot(pose, { base: [185, 420], color: COLORS.measurement, label: "q(t)" })}
    ${robot(pose, { base: [780, 420], color: COLORS.seal, splats: true, label: "G(q(t))" })}
    <g opacity="${opacity(phase(progress, .05, .24))}">
      <rect x="370" y="93" width="190" height="84" rx="8" fill="#fff" stroke="${COLORS.line}"/>
      <text x="390" y="121" class="svg-label">OBSERVATION PACKET</text>
      <text x="390" y="149" class="svg-formula">{ I(t), q(t), timestamp }</text>
    </g>
    ${arrow(344, 118, 368, 135, frameProgress)}${arrow(562, 135, 704, 250, frameProgress)}
    <circle cx="${packetX}" cy="135" r="8" fill="${COLORS.gold}" stroke="#fff" stroke-width="3"/>
    <path d="M95 470H855" stroke="${COLORS.dun}" stroke-width="7" stroke-linecap="round"/>
    <path d="M95 470H${lerp(95,855,progress)}" stroke="${COLORS.seal}" stroke-width="7" stroke-linecap="round"/>
    <text x="475" y="495" text-anchor="middle" class="svg-small">appearance, articulation, and time advance together</text>`;
}

function sceneFailure(progress) {
  const fit = phase(progress, .05, .88);
  const fitted = interpolatePose(POSES.measuredPose, POSES.stalePose, fit);
  const lossPath = pathFromPoints(sampleCurve(t => .95 * Math.exp(-4.2 * t) + .06, 485, 105, 390, 120));
  const errorPath = pathFromPoints(sampleCurve(t => .1 + .87 * (1 - Math.exp(-3.2 * t)), 485, 292, 390, 120));
  return `${backdrop()}
    <text x="55" y="48" class="svg-label">STALE VIEW FIT</text>
    ${camera(70, 72, `camera t − ${Math.round(84 * fit)} ms`, fit)}
    ${robot(POSES.measuredPose, { base: [245, 425], color: COLORS.measurement, opacity: .72, dashed: true, label: "physical state q*" })}
    ${robot(fitted, { base: [245, 425], color: COLORS.seal, splats: true, label: "fitted state q+" })}
    <path d="M132 126Q205 145 252 228" fill="none" stroke="${COLORS.danger}" stroke-width="3" stroke-dasharray="8 6"/>
    <text x="485" y="82" class="svg-label">VISUAL RESIDUAL</text>
    <rect x="470" y="91" width="420" height="151" rx="8" fill="#fff" stroke="${COLORS.line}"/>
    <path d="${lossPath}" fill="none" stroke="${COLORS.success}" stroke-width="4" pathLength="1" stroke-dasharray="1" stroke-dashoffset="${1 - fit}"/>
    <text x="852" y="122" class="svg-number">${(1 - .67 * fit).toFixed(2)}</text>
    <text x="485" y="269" class="svg-label">JOINT-STATE ERROR</text>
    <rect x="470" y="278" width="420" height="151" rx="8" fill="#fff" stroke="${COLORS.line}"/>
    <path d="${errorPath}" fill="none" stroke="${COLORS.danger}" stroke-width="4" pathLength="1" stroke-dasharray="1" stroke-dashoffset="${1 - fit}"/>
    <text x="824" y="406" class="svg-number">${(.31 + 4.763 * fit).toFixed(2)}°</text>
    <g opacity="${opacity(phase(progress,.66,.86))}"><rect x="322" y="450" width="330" height="42" rx="7" fill="${COLORS.soft}" stroke="${COLORS.danger}"/><text x="487" y="476" text-anchor="middle" class="svg-title">Better image fit ≠ better robot state</text></g>`;
}

function sceneContract(progress) {
  const stages = [
    [55, "Measurement", "qᵐ"], [270, "Bounded proposal", "q+"], [485, "Separate evidence", "e"], [700, "State authority", "Aτ(e)"]
  ];
  const active = clamp(Math.floor(progress * 4), 0, 3);
  const x = lerp(120, 765, ease(progress));
  const branch = phase(progress, .68, .88);
  const fallback = phase(progress, .86, 1);
  return `${backdrop()}
    ${stages.map(([sx,title,symbol],index) => roundedCard(sx, 178, 175, 100, title, symbol, index === 3 ? COLORS.seal : COLORS.gold, index <= active)).join("")}
    ${arrow(230,228,270,228,phase(progress,.12,.28))}${arrow(445,228,485,228,phase(progress,.34,.50))}${arrow(660,228,700,228,phase(progress,.56,.68))}
    <circle cx="${x}" cy="228" r="9" fill="${COLORS.gold}" stroke="#fff" stroke-width="3"/>
    <path d="M787 278Q808 336 846 350" fill="none" stroke="${COLORS.success}" stroke-width="4" stroke-dasharray="${branch * 95} 110" marker-end="url(#arrow)"/>
    <path d="M740 278Q706 350 650 368" fill="none" stroke="${COLORS.cadet}" stroke-width="4" stroke-dasharray="${fallback * 120} 130"/>
    <g opacity="${opacity(branch)}">${roundedCard(765, 348, 155, 72, "Verified update", "commit q+", COLORS.success, true)}</g>
    <g opacity="${opacity(fallback)}">${roundedCard(510, 365, 175, 72, "Fallback", "preserve qᵐ", COLORS.cadet, true)}</g>
    <text x="480" y="83" text-anchor="middle" class="svg-title">State estimation and state-changing authority are separate decisions.</text>
    <text x="480" y="112" text-anchor="middle" class="svg-formula">qˢ = Aτ(e)q+ + [1 − Aτ(e)]qᵐ</text>`;
}

function sceneSpatial(progress) {
  const proposal = interpolatePose(POSES.measuredPose, POSES.proposalPose, phase(progress,.18,.55));
  const scores = evidenceScores(phase(progress,.35,.76));
  const commit = phase(progress,.70,.92);
  const finalPose = interpolatePose(POSES.measuredPose, POSES.proposalPose, commit);
  return `${backdrop()}
    ${camera(92, 56, "head view", progress)}${camera(332, 56, "wrist view", 1-progress)}
    <path d="M155 117L440 230M395 117L440 230" stroke="${COLORS.cadet}" stroke-width="2" stroke-dasharray="6 5" opacity=".7"/>
    ${robot(POSES.measuredPose, { base: [440, 438], color: COLORS.measurement, opacity: .55, dashed: true, label: "measurement" })}
    ${robot(proposal, { base: [440, 438], color: COLORS.gold, opacity: 1-commit*.65, splats: true, label: "proposal" })}
    ${robot(finalPose, { base: [440, 438], color: COLORS.success, opacity: commit, splats: true, label: "verified state" })}
    <g transform="translate(670 92)">
      <rect width="235" height="250" rx="9" fill="#fff" stroke="${COLORS.line}" filter="url(#soft-shadow)"/>
      <text x="18" y="29" class="svg-label">CROSS-VIEW EVIDENCE</text>
      ${gauge(18,60,150,"visual gain r",scores.visualGain)}
      ${gauge(18,105,150,"gradient support g",scores.gradient)}
      ${gauge(18,150,150,"direction support a",scores.direction)}
      ${gauge(18,195,150,"state agreement d",scores.agreement)}
      <rect x="18" y="222" width="199" height="18" rx="9" fill="${commit > .5 ? COLORS.success : COLORS.dun}"/>
      <text x="117" y="235" text-anchor="middle" fill="#fff" font-size="10" font-weight="800">${commit > .5 ? "WHOLE-STATE COMMIT" : "COLLECTING EVIDENCE"}</text>
    </g>
    <path d="M575 312H645" stroke="${COLORS.gold}" stroke-width="3" marker-end="url(#arrow)"/>
    <text x="610" y="300" text-anchor="middle" class="svg-small">verify complete proposal</text>`;
}

function sceneTemporal(progress) {
  const align = phase(progress,.16,.78);
  const lag = .11 * (1-align);
  const image = pathFromPoints(sampleCurve(t => motionSignal(t,0), 70, 95, 555, 120));
  const joint = pathFromPoints(sampleCurve(t => motionSignal(t,lag), 70, 95, 555, 120));
  const corr = pathFromPoints(sampleCurve(t => .18 + .78*Math.exp(-Math.pow((t-.5)/(0.16+.12*(1-align)),2)), 70, 300, 555, 100));
  const markerX = lerp(570,347,align);
  return `${backdrop()}
    <text x="70" y="68" class="svg-label">CROSS-MODAL MOTION</text>
    <rect x="55" y="78" width="590" height="160" rx="8" fill="#fff" stroke="${COLORS.line}"/>
    <path d="${image}" fill="none" stroke="${COLORS.seal}" stroke-width="4"/>
    <path d="${joint}" fill="none" stroke="${COLORS.gold}" stroke-width="4"/>
    <g transform="translate(470 92)"><circle r="5" fill="${COLORS.seal}"/><text x="12" y="4" class="svg-small">image motion</text><circle cx="0" cy="24" r="5" fill="${COLORS.gold}"/><text x="12" y="28" class="svg-small">joint motion</text></g>
    <text x="70" y="277" class="svg-label">LAG CORRELATION</text>
    <rect x="55" y="287" width="590" height="137" rx="8" fill="#fff" stroke="${COLORS.line}"/>
    <path d="${corr}" fill="none" stroke="${COLORS.cadet}" stroke-width="4"/>
    <line x1="347" y1="300" x2="347" y2="408" stroke="${COLORS.success}" stroke-width="2" stroke-dasharray="5 5"/>
    <circle cx="${markerX}" cy="${lerp(370,315,align)}" r="9" fill="${COLORS.gold}" stroke="#fff" stroke-width="3"/>
    <text x="${markerX}" y="${lerp(395,340,align)}" text-anchor="middle" class="svg-small">${Math.round(56*(1-align))} ms</text>
    ${roundedCard(700, 292, 205, 76, "Lag proposal", `Δt+ = ${Math.round(56*(1-align))} ms`, COLORS.gold, true)}
    ${roundedCard(700, 401, 205, 76, align>.72 ? "Verified resampling" : "Timing evidence", align>.72 ? "publish q(t + Δt+)" : "peak · separation · motion", align>.72 ? COLORS.success : COLORS.cadet, true)}
    ${arrow(802,369,802,400,align)} `;
}

function sceneControlled(progress) {
  const t = ease(progress);
  const pose = interpolatePose(POSES.measuredPose, POSES.proposalPose, t);
  const dots = Array.from({length:48},(_,i) => {
    const col=i%12,row=Math.floor(i/12); const accepted = i%4===0;
    return `<circle cx="${84+col*35}" cy="${90+row*34}" r="8" fill="${accepted?COLORS.success:COLORS.cadet}" opacity="${.32+.68*phase(progress,i/70,(i/70)+.2)}"/>`;
  }).join("");
  const measuredWidth=230, syncWidth=lerp(230,126,t);
  return `${backdrop()}
    <text x="62" y="48" class="svg-label">48 HELD-OUT VIEW / OFFSET CONDITIONS</text>
    <g>${dots}</g>
    <text x="82" y="241" class="svg-small">green: helpful proposal committed</text><text x="82" y="260" class="svg-small">gray: measurement preserved</text>
    <g transform="translate(60 300)"><text y="-15" class="svg-label">WHOLE-STATE qMAE</text><text x="0" y="24" class="svg-small">Measured</text><rect x="76" y="8" width="${measuredWidth}" height="20" rx="4" fill="${COLORS.measurement}"/><text x="${86+measuredWidth}" y="24" class="svg-number">0.590°</text><text x="0" y="70" class="svg-small">KineSync-GS</text><rect x="76" y="54" width="${syncWidth}" height="20" rx="4" fill="${COLORS.seal}"/><text x="${86+syncWidth}" y="70" class="svg-number">${lerp(.590,.322,t).toFixed(3)}°</text></g>
    ${robot(POSES.measuredPose,{base:[705,440],color:COLORS.measurement,opacity:.42,dashed:true})}
    ${robot(pose,{base:[705,440],color:COLORS.success,splats:true,label:"verified recovery"})}
    <g opacity="${opacity(phase(progress,.55,.85))}"><rect x="650" y="72" width="248" height="76" rx="8" fill="#fff" stroke="${COLORS.success}" stroke-width="2"/><text x="774" y="104" text-anchor="middle" class="svg-number">12 / 12 helpful retained</text><text x="774" y="129" text-anchor="middle" class="svg-title">0 / 48 harmful updates</text></g>`;
}

function sceneMechanism(progress) {
  const t=ease(progress);
  const dual=lerp(42.86,80.95,t), retained=lerp(61.11,88.89,t);
  return `${backdrop()}
    <text x="55" y="45" class="svg-label">FACTORIZED CALIBRATION</text>
    ${roundedCard(55,70,190,84,"Shared joint zero","persistent across views",COLORS.seal,true)}
    ${roundedCard(300,70,180,84,"Camera A residual","view-specific",COLORS.cadet,true)}
    ${roundedCard(535,70,180,84,"Camera B residual","view-specific",COLORS.cadet,true)}
    ${arrow(245,112,298,112,phase(progress,.08,.25))}${arrow(480,112,533,112,phase(progress,.22,.38))}
    <path d="M150 155Q270 210 390 224M625 155Q510 210 390 224" fill="none" stroke="${COLORS.gold}" stroke-width="3"/>
    ${roundedCard(285,225,215,80,"Cross-view evidence","calibrated before evaluation",COLORS.success,t>.28)}
    <g transform="translate(65 350)"><text y="-18" class="svg-label">DUAL-VIEW RECOVERY</text><rect width="340" height="30" rx="5" fill="#ece6e0"/><rect width="${340*dual/100}" height="30" rx="5" fill="${COLORS.seal}"/><text x="${Math.min(325,340*dual/100+10)}" y="22" class="svg-number">${dual.toFixed(2)}%</text></g>
    <g transform="translate(520 350)"><text y="-18" class="svg-label">SUCCESSFUL PROPOSALS RETAINED</text><rect width="340" height="30" rx="5" fill="#ece6e0"/><rect width="${340*retained/100}" height="30" rx="5" fill="${COLORS.gold}"/><text x="${Math.min(325,340*retained/100+10)}" y="22" class="svg-number">${retained.toFixed(2)}%</text></g>
    <text x="480" y="449" text-anchor="middle" class="svg-title">Stronger evidence improves the accepted update set.</text>`;
}

function sceneRealWorld(progress) {
  const t=ease(progress);
  const pose=interpolatePose(POSES.stalePose,POSES.proposalPose,t);
  const packet=lerp(160,650,t);
  return `${backdrop()}
    <text x="355" y="47" class="svg-label">REAL-WORLD PIPER STATE STREAM</text>
    ${camera(360,68,"head",progress)}${camera(545,68,"wrist",1-progress)}
    ${roundedCard(335,180,160,74,"Component-GS","state-indexed replay",COLORS.gold,t>.18)}
    ${roundedCard(535,180,160,74,"Cross-view gate","verify proposal",COLORS.success,t>.45)}
    ${arrow(420,142,420,178,phase(progress,.08,.25))}${arrow(605,142,605,178,phase(progress,.08,.25))}${arrow(495,217,533,217,phase(progress,.30,.55))}
    <circle cx="${packet}" cy="290" r="9" fill="${COLORS.gold}" stroke="#fff" stroke-width="3"/>
    <path d="M320 290H720" stroke="${COLORS.dun}" stroke-width="7" stroke-linecap="round"/>
    <path d="M320 290H${packet}" stroke="${COLORS.seal}" stroke-width="7" stroke-linecap="round"/>
    ${robot(pose,{base:[760,440],color:COLORS.success,splats:true,label:"published qˢ(t)"})}
    <g transform="translate(332 330)">${gauge(0,0,260,"state confidence",lerp(.35,.96,t),COLORS.success)}${gauge(0,50,260,"cross-view support",lerp(.28,.94,t),COLORS.gold)}${gauge(0,100,260,"stream availability",lerp(.62,.99,t),COLORS.cadet)}</g>`;
}

function sceneOnline(progress) {
  const t=ease(progress);
  const lanes=[["RGB-D frames",88],["joint telemetry",178],["kinematic prior",268]];
  return `${backdrop()}
    <text x="55" y="45" class="svg-label">ONLINE SYNCHRONIZATION CHAIN</text>
    ${lanes.map(([name,y],i)=>`${roundedCard(55,y,170,60,name,i===0?"asynchronous":i===1?"high-rate state":"fixed model",i===0?COLORS.seal:COLORS.cadet,true)}${arrow(225,y+30,315,y+30,phase(progress,.04+i*.05,.28+i*.05))}`).join("")}
    ${roundedCard(315,104,180,90,"Temporal verify","associate camera + state",COLORS.gold,t>.18)}
    ${roundedCard(315,230,180,90,"Spatial verify","recover + gate q+",COLORS.success,t>.38)}
    <path d="M405 194V228" stroke="${COLORS.gold}" stroke-width="3" marker-end="url(#arrow)"/>
    ${roundedCard(570,285,205,82,"Synchronized state","qˢ(t), timestamp, evidence",COLORS.seal,t>.62)}
    ${arrow(495,275,568,326,phase(progress,.50,.74))}
    <g transform="translate(560 415)">${gauge(0,0,220,"recovery",lerp(.4272,.8305,t),COLORS.success)}${gauge(0,45,220,"harmful update",lerp(.0668,.0036,t),COLORS.danger)}</g>
    <g opacity="${opacity(phase(progress,.68,.9))}"><circle cx="850" cy="392" r="48" fill="#fff" stroke="${COLORS.success}" stroke-width="4"/><text x="850" y="386" text-anchor="middle" class="svg-number">3.07</text><text x="850" y="408" text-anchor="middle" class="svg-small">mrad qMAE</text></g>`;
}

function scenePolicy(progress) {
  const t=ease(progress);
  const raw=58.33, sync=lerp(58.33,71.67,t);
  const taskDots=Array.from({length:24},(_,i)=>`<circle cx="${648+(i%8)*28}" cy="${112+Math.floor(i/8)*30}" r="7" fill="${i<Math.round(24*sync/100)?COLORS.success:COLORS.dun}" opacity="${.3+.7*phase(progress,i/35,i/35+.2)}"/>`).join("");
  return `${backdrop()}
    ${roundedCard(365,72,235,78,"Unchanged policy π","same weights · same episode budget",COLORS.seal,true)}
    ${roundedCard(645,72,205,78,"Action chunk","generated from current state",COLORS.gold,t>.24)}
    ${arrow(600,111,643,111,phase(progress,.12,.35))}
    <path d="M275 94H363" stroke="${COLORS.measurement}" stroke-width="4" marker-end="url(#arrow-seal)"/><text x="275" y="80" class="svg-label">RAW STATE</text>
    <path d="M275 133H363" stroke="${COLORS.success}" stroke-width="4" marker-end="url(#arrow-seal)"/><text x="275" y="162" class="svg-label">SYNCHRONIZED STATE</text>
    <g transform="translate(345 352)"><text y="-17" class="svg-label">TASK-MACRO SUCCESS</text><text x="0" y="21" class="svg-small">raw</text><rect x="74" y="6" width="${raw*2.55}" height="24" rx="4" fill="${COLORS.measurement}"/><text x="${84+raw*2.55}" y="24" class="svg-title">58.33%</text><text x="0" y="68" class="svg-small">KineSync</text><rect x="74" y="53" width="${sync*2.55}" height="24" rx="4" fill="${COLORS.seal}"/><text x="${84+sync*2.55}" y="71" class="svg-title">${sync.toFixed(2)}%</text></g>
    <text x="680" y="205" class="svg-label">PAIRED REAL-ROBOT EPISODES</text><g transform="translate(0 128)">${taskDots}</g>
    <text x="760" y="362" text-anchor="middle" class="svg-small">240 paired episodes across three tasks</text>
    ${robot(interpolatePose(POSES.measuredPose,POSES.proposalPose,t),{base:[845,468],color:COLORS.success,splats:true,label:"same policy, better state"})}`;
}

function sceneClosing(progress) {
  const t=ease(progress);
  const nodes=[[70,105,"Physical robot","camera + telemetry"],[70,330,"Gaussian twin","articulated renderer"],[700,105,"Simulator","state adapter"],[700,330,"Robot policy","action consumer"]];
  return `${backdrop()}
    <circle cx="480" cy="255" r="96" fill="#fff" stroke="${COLORS.seal}" stroke-width="4" filter="url(#soft-shadow)"/>
    <circle cx="480" cy="255" r="72" fill="${COLORS.softGold || '#f5eddf'}" stroke="${COLORS.gold}" stroke-width="2"/>
    <text x="480" y="235" text-anchor="middle" class="svg-label">KINESYNC-GS</text><text x="480" y="266" text-anchor="middle" class="svg-number">qˢ(t)</text><text x="480" y="291" text-anchor="middle" class="svg-small">verified state bus</text>
    ${nodes.map(([x,y,title,body],i)=>`<g>${roundedCard(x,y,190,78,title,body,i<2?COLORS.cadet:COLORS.gold,phase(progress,i*.08,.45+i*.08)>.1)}<path d="${x<480?`M${x+190} ${y+39}L384 ${255+(y-255)*.15}`:`M576 ${255+(y-255)*.15}L${x} ${y+39}`}" fill="none" stroke="${i%2?COLORS.gold:COLORS.cadet}" stroke-width="3" stroke-dasharray="${t*220} 230"/></g>`).join("")}
    <g opacity="${opacity(phase(progress,.55,.9))}"><text x="480" y="476" text-anchor="middle" class="svg-title">One proposal–evidence–update contract across robot stacks.</text></g>`;
}

const renderers = {
  real2sim2real: sceneReal2Sim,
  failure: sceneFailure,
  contract: sceneContract,
  spatial: sceneSpatial,
  temporal: sceneTemporal,
  controlled: sceneControlled,
  mechanism: sceneMechanism,
  "real-world": sceneRealWorld,
  online: sceneOnline,
  policy: scenePolicy,
  closing: sceneClosing,
};

export function renderScene(id, progress) {
  return (renderers[id] || sceneContract)(clamp(progress));
}

export { sceneState };
