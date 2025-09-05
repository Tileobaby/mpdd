## Dual-Arm Planning: From Single to Dual Robots (CHOMP/GPMP2)

### Goal
Enable motion planning for two Panda arms in the same 3D environment without changing CHOMP/GPMP2 cores. The trick: model two robots as a single "composite" robot, and add an inter-robot collision cost.

### Key Additions (no core algorithm changes)
- Composite robot:
  - File: `deps/torch_robotics/torch_robotics/robots/composite_panda.py`
  - Exported via: `deps/torch_robotics/torch_robotics/robots/__init__.py`
  - Concatenates two Panda configs; concatenates FK collision points; supports per-robot base translation.
- Inter-robot collision cost (example-level):
  - Simple nearest-point penalty: `penalty = relu(margin - min_pairwise_distance)`.
  - Implemented inline in examples as a lightweight `InterRobotCollisionField`.
- Examples:
  - CHOMP: `deps/motion_planning_baselines/examples/dual_panda_spheres_CHOMP.py`
  - GPMP2: `deps/motion_planning_baselines/examples/dual_panda_spheres_GPMP.py`

### How It Works
- Joint space: `q = [q_robot1, q_robot2]`, limits are concatenated.
- FK for collision: run each Panda separately, then concatenate link points.
- Costs used by planners:
  - Environment/object collisions, self-collision, workspace boundaries from `PlanningTask`.
  - Inter-robot collision (new) added to the cost set.
- Planners (CHOMP/GPMP2) run unmodified; just pass the composite robot and the extended cost set.

### File Map
- Composite robot: `torch_robotics/robots/composite_panda.py`
- Dual-arm CHOMP example: `motion_planning_baselines/examples/dual_panda_spheres_CHOMP.py`
- Dual-arm GPMP2 example: `motion_planning_baselines/examples/dual_panda_spheres_GPMP.py`

### Quick Start
CHOMP:
```bash
cd deps/motion_planning_baselines/examples
python dual_panda_spheres_CHOMP.py
```

GPMP2:
```bash
cd deps/motion_planning_baselines/examples
python dual_panda_spheres_GPMP.py
```

Outputs: console stats, MP4 animations (joint-space and trajectories), and pickled results.

### Important Parameters / Tips
- Base translations: shift robot 2 (e.g., `x = 0.6`) to reduce initial inter-robot collisions.
- Inter-robot margin: try `0.03–0.06`. Larger = safer but may shrink feasible space.
- Cost weights: give inter-robot collision slightly higher weight than environment collision (e.g., 15 vs 10) to prioritize mutual avoidance.
- Discretization: increase `n_support_points` for finer collision checking (more compute).
- CHOMP: tune `step_size`, `opt_iters`, `grad_clip`.
- GPMP2: tune `sigma_*`, `opt_iters`, `num_samples`; ensure `sigma_start_sample` and `sigma_goal_sample` are set (non-None).

### Notes / Limitations
- Example inter-robot cost uses nearest-link distance without per-link radii; can be upgraded to per-point SDF-style margins.
- Start/goal sampling checks inter-robot collision at single configurations; stricter checks could validate along interpolated paths.


