# KineSync-GS project website

This directory is a zero-build GitHub Pages site.

- `index.html`: main project page.
- `explainer.html`: 150-second interactive method animation with arbitrary seeking, chapter navigation, English subtitles, and narration-ready timing.
- `static/data/explainer-timeline.json`: single source of truth for scene boundaries, media, metrics, and subtitles.
- `NARRATION.md`: recording script and audio handoff contract.

```bash
python -m http.server 8000 --directory docs
```

GitHub Pages deployment source: `main` branch, `/docs` directory.

Preview the interactive demonstration at `http://localhost:8000/explainer.html`.
