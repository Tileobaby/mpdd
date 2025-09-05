## 从单机器人到双机器人：实现说明与改动总结

### 背景与目标
- 在不修改 CHOMP/GPMP2 算法核心实现的前提下，支持两台 Panda 机械臂在同一 3D 球体环境中进行联合轨迹优化，要求同时避开环境与彼此（含各自自碰）。
- 方案：抽象“联合机器人”，让优化器视为一个更高维的单机器人；在代价中新增“跨机器人碰撞”项。

---

## 关键改动总览

- 新增“联合机器人”组件：
  - 文件：`deps/torch_robotics/torch_robotics/robots/composite_panda.py`
  - 导出：`deps/torch_robotics/torch_robotics/robots/__init__.py`
  - 作用：将两台 `RobotPanda` 拼为一个“联合机器人”，用于 CHOMP/GPMP2 统一优化。

- 新增示例（双 Panda + CHOMP）：
  - 文件：`deps/motion_planning_baselines/examples/dual_panda_spheres_CHOMP.py`
  - 内容：构造两台 Panda + `CompositePandaRobot`，加入跨机器人碰撞代价，CHOMP 优化与可视化。

- 新增示例（双 Panda + GPMP2）：
  - 文件：`deps/motion_planning_baselines/examples/dual_panda_spheres_GPMP.py`
  - 内容：同上，适配 GPMP2 的代价构造与参数，加入跨机器人碰撞代价与收敛判据。

- 兼容性修复与稳定性改进：
  - 修复 `EnvSpheres3D.get_gpmp2_params` 中 `isinstance(robot, RobotPanda, RobotFanuc)` 的语法错误为 `isinstance(robot, (RobotPanda, RobotFanuc))`。
  - 修复跨机器人代价中的最小值归约（避免 `torch.min` 的 tuple 维参数用法）。
  - 为两台机器人引入基座平移（base translation），减少初始互碰采样失败。
  - GPMP2 示例补齐 `sigma_start_sample`/`sigma_goal_sample` 以避免 `None ** 2` 错误。

---

## 代码结构与新增文件

- 联合机器人
  - `torch_robotics/robots/composite_panda.py`：新增 `CompositePandaRobot`
  - `torch_robotics/robots/__init__.py`：导出 `CompositePandaRobot`

- 示例脚本
  - `motion_planning_baselines/examples/dual_panda_spheres_CHOMP.py`
  - `motion_planning_baselines/examples/dual_panda_spheres_GPMP.py`

- 兼容性修复
  - `torch_robotics/environments/env_spheres_3d.py`：`isinstance` 参数修复

---

## 实现细节

### 1) CompositePandaRobot（联合机器人）
- 设计理念：把两台 Panda 的关节向量拼接成一个“大关节”；FK 碰撞点也拼接；优化器统一处理更高维的轨迹。
- 关键点：
  - 关节拼接：`q = [q_panda_1, q_panda_2]`，`q_limits` 同样拼接。
  - 碰撞 FK：`fk_map_collision(q)` 内部拆分 `q1,q2`，分别调各自 `fk_map_collision`，然后在“链接点”维度拼接返回。
  - 环境/边界代价元数据：
    - 合并两台机器人的 `link_idxs_for_object_collision_checking` 和 `link_margins_for_object_collision_checking_tensor`，确保对“所有链接点”生效。
  - 自碰（self-collision）：
    - 使用内部包装器 `_CombinedSelfCollisionField` 分别计算两台 Panda 的自碰代价并相加（不含“跨机器人”自碰）。
  - 基座平移（base translation）：
    - 支持为每台机器人设置 `base_translation_[1|2]`（世界系平移），在 FK 输出上叠加，用于在同一环境内拉开两台机械臂的初始位置，提升“无互碰”采样成功率。

### 2) 跨机器人碰撞代价（InterRobotCollisionField）
- 实现于示例中（两个文件里各自定义了简单版本）：
  - 输入：联合机器人 `link_pos`（形状 `(B,H,T,3)`），按两台 Panda 的链接数切分为 `pos_r1`/`pos_r2`。
  - 计算：两两点对欧氏距离 `dists = ||pos_r1 - pos_r2||`，对链接维做最小化得到 `min_dist(B,H)`。
  - 代价：`penalty = relu(margin - min_dist)`，低于安全裕度则惩罚。
- 说明：
  - 这是一个简化且可导的“最近点距离”惩罚。若需更物理真实，可结合各链接点半径（或 margin）做逐点可变安全距离。

### 3) CHOMP 适配（dual_panda_spheres_CHOMP.py）
- 合成联合机器人与任务：
  - 创建 `RobotPanda` x2 → `CompositePandaRobot`（给第二台基座平移 `x=0.6`）→ 单一 `PlanningTask`。
- 起终点采样：
  - 先用 `task.random_coll_free_q(n_samples=2)` 获取联合配置（不含跨机器人判定），再用 `InterRobotCollisionField` 检测 start/goal 的互碰（失败则重采）；
  - 增大重试次数，适当降低 `inter margin`，提升成功率。
- 代价构造：
  - 取 `task.get_collision_fields()`（含自碰、环境、边界），再追加 `InterRobotCollisionField`；
  - 放入 `CostComposite`，对跨机器人代价给略高权重（如 15.0）以强化互相避碰。
- 优化与可视化：
  - `CHOMP` 的核心未改动（`n_dof` 改为联合自由度），输出联合轨迹，`PlanningVisualizer` 直接渲染联合机器人（内部分别画两台）。

### 4) GPMP2 适配（dual_panda_spheres_GPMP.py）
- 联合机器人与任务同上，跨机器人代价加入 `collision_fields`，通过 `build_gpmp2_cost_composite` 自动装配到 `CostComposite`。
- 参数注意：
  - 手动给出 `sigma_start/sample/goal/gp` 等（源自单机 Panda 的合理默认），并补齐 `sigma_start_sample`/`sigma_goal_sample` 以避免 `None`。
  - `n_support_points` 设为 128；`opt_iters` 设为 120；可结合收敛判据提前停止。
- 优化与可视化与 CHOMP 示例保持一致的接口与输出。

---

## 运行与验证

- CHOMP（双 Panda）：
```bash
cd deps/motion_planning_baselines/examples
python dual_panda_spheres_CHOMP.py
```

- GPMP2（双 Panda）：
```bash
cd deps/motion_planning_baselines/examples
python dual_panda_spheres_GPMP.py
```

- 输出：
  - 控制台打印统计信息（成功率/碰撞比例/时间）。
  - 生成若干 mp4（关节空间迭代/机器人轨迹）与 pickle 结果文件，文件名基于脚本名。

---

## 参数建议与调优

- 基座平移：适当拉开两台机械臂（如第二台 `x=0.6~0.8`），可显著提升初始采样成功率。
- 跨机器人裕度（margin）：可从 `0.03~0.06` 间试探；偏大更保守但可能使可行域过小。
- 代价权重：跨机器人碰撞项权重可略高于环境碰撞（如 15 vs 10），避免彼此挤压。
- 优化步长与次数：CHOMP 的 `step_size/opt_iters/grad_clip` 与 GPMP2 的 `sigma_*`、`opt_iters`、`n_support_points` 可按收敛与安全性权衡调整。
- 插值密度：`n_support_points` 越大碰撞评估越精细，但计算量增大。

---

## 已知限制与后续优化

- 跨机器人代价目前为“最小点对距离”惩罚，未显式使用每个链接点的半径。可升级为基于每点 margin 的逐点 SDF 逻辑，精度更高。
- 当前示例只在位置 FK 点上加惩罚，若需要手爪或抓取物体的精细模型，可进一步扩展点集。
- Sampling 阶段的互碰检查只对 start/goal 单点做检测（H=1），如需更严格，可检查插值线性路径或 GP 插值路径。

---

## 快速迁移清单（Checklist）

- 机器人层
  - 用 `RobotPanda` x2 创建 `CompositePandaRobot`（可设置 `base_translation_2`）；
  - 用联合机器人 + 环境构造单一 `PlanningTask`。

- 代价层
  - 取 `task.get_collision_fields()`；
  - 新增 `InterRobotCollisionField`，加入 `collision_fields` 或 `CostComposite`。

- 优化器层
  - CHOMP/GPMP2 核心不变：`n_dof = composite_robot.q_dim`，`start/goal = cat(start1, start2)`；
  - CHOMP：`CostComposite` 直接传；
  - GPMP2：将 `collision_fields`、`sigma_*` 参数传给 `GPMP2`（内部由 `build_gpmp2_cost_composite` 组装）。

- 采样与可视化
  - 采样联合 start/goal 后做互碰过滤（必要时增加重试/放宽 margin/增大基座偏移）；
  - `PlanningVisualizer` 直接渲染联合机器人（内部分别绘制两台）。


