# Panda presentation model

The browser loads `panda.urdf` using URDFLoader, retaining Panda's joint origins,
axes, limits, hand transform and finger mimic joint. Each visual link is a GLB
export of the corresponding detailed mesh parts from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda).
The model's Apache-2.0 license is included here. `mesh-export.json` lists the
original mesh parts and materials used in each exported link.

Presentation adaptation: collision geometry is omitted; visual meshes are
converted from OBJ to GLB and referenced with local relative paths. This is an
articulated visualization, not a browser physics benchmark.

`tour-robot.mjs` samples deterministic elliptical Gaussians on the visual
surfaces. Their centers and tangent axes remain in the visual's local frame, so
the URDF hierarchy moves them with the corresponding link. This teaches the
link-binding mechanism; the sampled Gaussians are not a learned reconstruction.
Measured paper evidence is provided by the separately identified recordings
and result figures in the narration manifest.
