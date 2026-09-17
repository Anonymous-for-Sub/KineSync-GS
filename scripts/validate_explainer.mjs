import { access, readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = path.resolve(process.cwd(), "docs");
const manifestPath = path.join(root, "static/data/explainer-timeline.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));

const errors = [];
const required = ["id", "start", "end", "title", "shortTitle", "type", "media", "metric", "detail", "caption"];

if (!Array.isArray(manifest.scenes) || manifest.scenes.length === 0) errors.push("timeline has no scenes");
if (manifest.audio !== null) errors.push("audio must remain null until the supplied recording is measured");
if (manifest.timingSource !== "visual-plan-pre-audio") errors.push("pre-audio timing source is not explicit");

let cursor = 0;
for (const [index, scene] of manifest.scenes.entries()) {
  for (const key of required) if (!(key in scene)) errors.push(`scene ${index + 1} is missing ${key}`);
  if (scene.start !== cursor) errors.push(`scene ${scene.id} starts at ${scene.start}, expected ${cursor}`);
  if (!(scene.end > scene.start)) errors.push(`scene ${scene.id} has invalid bounds`);
  cursor = scene.end;
  if (!scene.caption?.en || !scene.caption?.zh) errors.push(`scene ${scene.id} lacks bilingual subtitles`);
  for (const field of ["media", "poster", "secondary"]) {
    if (!scene[field]) continue;
    const file = path.join(root, scene[field]);
    try { await access(file); } catch { errors.push(`scene ${scene.id} references missing ${field}: ${scene[field]}`); }
  }
}

if (cursor !== manifest.duration) errors.push(`timeline ends at ${cursor}, duration is ${manifest.duration}`);
if (manifest.duration !== 150) errors.push(`expected a 150-second demonstration, found ${manifest.duration}`);

const html = await readFile(path.join(root, "explainer.html"), "utf8");
for (const asset of ["static/css/explainer.css", "static/js/explainer.js", "static/data/explainer-timeline.json"]) {
  if (asset !== "static/data/explainer-timeline.json" && !html.includes(asset)) errors.push(`explainer.html does not load ${asset}`);
}

if (errors.length) {
  console.error(errors.map((error) => `ERROR: ${error}`).join("\n"));
  process.exit(1);
}

console.log(`OK: ${manifest.scenes.length} contiguous scenes cover ${manifest.duration} seconds with bilingual subtitles and valid media paths.`);
