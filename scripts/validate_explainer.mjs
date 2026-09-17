import { access, readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = path.resolve(process.cwd(), "docs");
const manifestPath = path.join(root, "static/data/explainer-timeline.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));

const errors = [];
const required = ["id", "start", "end", "title", "shortTitle", "kicker", "metric", "detail", "formula", "caption"];

if (!Array.isArray(manifest.scenes) || manifest.scenes.length === 0) errors.push("timeline has no scenes");
if (manifest.audio !== null) errors.push("audio must remain null until the supplied recording is measured");
if (manifest.timingSource !== "visual-plan-pre-audio") errors.push("pre-audio timing source is not explicit");

let cursor = 0;
for (const [index, scene] of manifest.scenes.entries()) {
  for (const key of required) if (!(key in scene)) errors.push(`scene ${index + 1} is missing ${key}`);
  if (scene.start !== cursor) errors.push(`scene ${scene.id} starts at ${scene.start}, expected ${cursor}`);
  if (!(scene.end > scene.start)) errors.push(`scene ${scene.id} has invalid bounds`);
  cursor = scene.end;
  if (typeof scene.caption !== "string" || !scene.caption.trim()) errors.push(`scene ${scene.id} lacks an English subtitle`);
  for (const field of ["media", "poster"]) {
    if (!scene.evidence?.[field]) continue;
    const file = path.join(root, scene.evidence[field]);
    try { await access(file); } catch { errors.push(`scene ${scene.id} references missing evidence ${field}: ${scene.evidence[field]}`); }
  }
}

if (cursor !== manifest.duration) errors.push(`timeline ends at ${cursor}, duration is ${manifest.duration}`);
if (manifest.duration !== 150) errors.push(`expected a 150-second demonstration, found ${manifest.duration}`);

const html = await readFile(path.join(root, "explainer.html"), "utf8");
for (const asset of ["static/css/explainer.css", "static/js/explainer.js", "static/data/explainer-timeline.json"]) {
  if (asset !== "static/data/explainer-timeline.json" && !html.includes(asset)) errors.push(`explainer.html does not load ${asset}`);
}

const narration = await readFile(path.join(root, "NARRATION.md"), "utf8");
const englishOnlySurface = `${html}\n${JSON.stringify(manifest)}\n${narration}`;
if (/[\u4e00-\u9fff]/u.test(englishOnlySurface)) errors.push("the public animation or narration handoff contains Chinese text");

if (errors.length) {
  console.error(errors.map((error) => `ERROR: ${error}`).join("\n"));
  process.exit(1);
}

console.log(`OK: ${manifest.scenes.length} contiguous scenes cover ${manifest.duration} seconds with English subtitles and valid media paths.`);
