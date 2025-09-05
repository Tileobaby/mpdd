import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mp_baselines.planners.chomp import CHOMP
from mp_baselines.planners.costs.cost_functions import CostComposite, CostCollision
from torch_robotics.environments.env_spheres_3d import EnvSpheres3D
from torch_robotics.robots import CompositePandaRobot
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer


allow_ops_in_compiled_graph()


class InterRobotCollisionField:
    """
    简单的跨机器人碰撞代价：
    - 输入 link_pos 为联合机器人在 (batch, horizon, links_total, 3) 的链接采样点
    - 将其按两台机器人拆分，计算两两点对的最小距离
    - 代价 = relu(margin - min_pairwise_distance)，沿时间求和/或逐时刻返回
    注意：这里用常数 margin 近似安全半径，若需更精细，可按每个链接点半径构造。
    """

    def __init__(self, robot_composite: CompositePandaRobot, margin: float = 0.05, tensor_args=None):
        self.robot_composite = robot_composite
        self.margin = margin
        self.tensor_args = tensor_args if tensor_args is not None else robot_composite.tensor_args

        # 估计两台机器人的链接数量，用于拆分 link_pos
        self._n_links_r1 = len(robot_composite.robot1.diff_panda.get_link_names())
        self._n_links_r2 = len(robot_composite.robot2.diff_panda.get_link_names())

    def compute_cost(self, q, link_pos, **kwargs):
        # 统一形状为 (B, H, T, 3)
        if link_pos.ndim == 3:  # (B, T, 3)
            link_pos = link_pos.unsqueeze(1)
        elif link_pos.ndim != 4:
            raise NotImplementedError

        B, H, T, D = link_pos.shape
        assert D == 3
        assert T >= (self._n_links_r1 + self._n_links_r2)

        pos_r1 = link_pos[..., :self._n_links_r1, :]
        pos_r2 = link_pos[..., self._n_links_r1:self._n_links_r1 + self._n_links_r2, :]

        # 计算两两距离 (B,H,L1,L2)
        # 利用广播： (B,H,L1,1,3) - (B,H,1,L2,3)
        diff = pos_r1.unsqueeze(-2) - pos_r2.unsqueeze(-3)
        dists = torch.linalg.norm(diff, dim=-1)  # (B,H,L1,L2)

        # 逐维归约，先对 L2 取 min，再对 L1 取 min，得到 (B,H)
        min_over_l2 = torch.min(dists, dim=-1)[0]      # (B,H,L1)
        min_dist = torch.min(min_over_l2, dim=-1)[0]   # (B,H)
        # 代价：小于 margin 则有惩罚
        penalty = torch.relu(self.margin - min_dist)  # (B,H)
        return penalty

    def zero_grad(self):
        # 本字段自身没有参数梯度需要清零
        return


if __name__ == "__main__":
    base_file_name = Path(os.path.basename(__file__)).stem

    seed = 2025
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {'device': device, 'dtype': torch.float32}

    # ---------------------------- Environment, Robots, Composite Task ---------------------------------
    env = EnvSpheres3D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=tensor_args
    )

    robot1 = RobotPanda(
        use_collision_spheres=True,
        use_self_collision_storm=False,
        tensor_args=tensor_args
    )

    robot2 = RobotPanda(
        use_collision_spheres=True,
        use_self_collision_storm=False,
        tensor_args=tensor_args
    )

    # 将第二台机器人沿 x 方向平移，以降低互相碰撞概率
    composite_robot = CompositePandaRobot(
        robot1, robot2,
        base_translation_1=torch.tensor([0.0, 0.0, 0.0], **tensor_args),
        base_translation_2=torch.tensor([0.6, 0.0, 0.0], **tensor_args),
        tensor_args=tensor_args
    )

    task = PlanningTask(
        env=env,
        robot=composite_robot,
        ws_limits=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], **tensor_args),
        obstacle_cutoff_margin=0.05,
        tensor_args=tensor_args
    )

    # -------------------------------- Start & Goal (joint concat) ---------------------------------
    # 简单方式：从联合任务中直接采两个无碰撞配置（注意不包含跨机器人碰撞），再用跨机器人检查过滤
    # 稍微降低互碰 margin，便于采样
    inter_field = InterRobotCollisionField(composite_robot, margin=0.04, tensor_args=tensor_args)

    def sample_dual_free_q(max_trials=1000):
        for _ in range(max_trials):
            q_free = task.random_coll_free_q(n_samples=2)
            start_state = q_free[0]
            goal_state = q_free[1]
            # 跨机器人起终点检查（逐单点，视为 H=1）
            lp = composite_robot.fk_map_collision(start_state.unsqueeze(0).unsqueeze(0))
            if inter_field.compute_cost(None, lp).max() > 0:
                continue
            lp = composite_robot.fk_map_collision(goal_state.unsqueeze(0).unsqueeze(0))
            if inter_field.compute_cost(None, lp).max() > 0:
                continue
            return start_state, goal_state
        raise RuntimeError("Failed to sample dual-robot non-inter-colliding start/goal")

    start_state = None
    goal_state = None
    start_state, goal_state = sample_dual_free_q()

    print(f'start_state (concat): {start_state}')
    print(f'goal_state  (concat): {goal_state}')

    multi_goal_states = goal_state.unsqueeze(0)

    duration = 5  # sec
    n_support_points = 64
    dt = duration / n_support_points

    # -------------------------------- Construct cost function ---------------------------------
    sigma_coll = 1e-3
    cost_collisions = []
    weights_cost_l = []
    for collision_field in task.get_collision_fields():
        cost_collisions.append(
            CostCollision(
                composite_robot, n_support_points,
                field=collision_field,
                sigma_coll=1.0,
                tensor_args=tensor_args
            )
        )
        weights_cost_l.append(10.0)

    # 跨机器人碰撞代价
    cost_collisions.append(
        CostCollision(
            composite_robot, n_support_points,
            field=inter_field,
            sigma_coll=1.0,
            tensor_args=tensor_args
        )
    )
    weights_cost_l.append(15.0)  # 略高，强化互相避碰

    cost_func_list = [*cost_collisions]
    cost_composite = CostComposite(
        composite_robot, n_support_points, cost_func_list,
        weights_cost_l=weights_cost_l,
        tensor_args=tensor_args
    )

    num_particles_per_goal = 10
    opt_iters = 60

    planner_params = dict(
        n_dof=composite_robot.q_dim,
        n_support_points=n_support_points,
        num_particles_per_goal=num_particles_per_goal,
        opt_iters=1,  # 保持 1 便于逐帧可视化
        dt=dt,
        start_state=start_state,
        cost=cost_composite,
        weight_prior_cost=1e-4,
        step_size=0.05,
        grad_clip=0.05,
        multi_goal_states=multi_goal_states,
        sigma_start_init=0.001,
        sigma_goal_init=0.001,
        sigma_gp_init=0.3,
        pos_only=False,
        tensor_args=tensor_args,
    )

    planner = CHOMP(**planner_params)

    # -------------------------------- Optimize ---------------------------------
    trajs_0 = planner.get_traj()
    trajs_iters = torch.empty((opt_iters + 1, *trajs_0.shape), **tensor_args)
    trajs_iters[0] = trajs_0
    with TimerCUDA() as t:
        for i in range(opt_iters):
            trajs = planner.optimize(debug=True)
            trajs_iters[i+1] = trajs
    print(f'Optimization time: {t.elapsed:.3f} sec, per iteration: {t.elapsed/opt_iters:.3f}')

    # -------------------------------- Save ---------------------------------
    trajs_iters_coll, trajs_iters_free = task.get_trajs_collision_and_free(trajs_iters[-1])
    results_data_dict = {
        'duration': duration,
        'n_support_points': n_support_points,
        'dt': dt,
        'trajs_iters_coll': trajs_iters_coll.unsqueeze(0) if trajs_iters_coll is not None else None,
        'trajs_iters_free': trajs_iters_free.unsqueeze(0) if trajs_iters_free is not None else None,
    }

    with open(os.path.join('./', f'{base_file_name}-results_data_dict.pickle'), 'wb') as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(
        task=task,
        planner=planner
    )

    print(f'----------------STATISTICS----------------')
    print(f'percentage free trajs: {task.compute_fraction_free_trajs(trajs_iters[-1])*100:.2f}')
    print(f'percentage collision intensity {task.compute_collision_intensity_trajs(trajs_iters[-1])*100:.2f}')
    print(f'success {task.compute_success_free_trajs(trajs_iters[-1])}')

    base_file_name = Path(os.path.basename(__file__)).stem

    pos_trajs_iters = composite_robot.get_position(trajs_iters)

    # 关节空间
    planner_visualizer.plot_joint_space_state_trajectories(
        trajs=trajs_iters[-1],
        pos_start_state=start_state, pos_goal_state=goal_state,
        vel_start_state=torch.zeros_like(start_state), vel_goal_state=torch.zeros_like(goal_state),
    )

    # 逐迭代动画
    planner_visualizer.animate_opt_iters_joint_space_state(
        trajs=trajs_iters,
        pos_start_state=start_state, pos_goal_state=goal_state,
        vel_start_state=torch.zeros_like(start_state), vel_goal_state=torch.zeros_like(goal_state),
        video_filepath=f'{base_file_name}-joint-space-opt-iters.mp4',
        n_frames=max((2, opt_iters // 10)),
        anim_time=5
    )

    # 机器人轨迹（Composite 会分别渲染两台机器人）
    planner_visualizer.render_robot_trajectories(
        trajs=pos_trajs_iters[-1], start_state=start_state, goal_state=goal_state,
        render_planner=False,
    )

    planner_visualizer.animate_robot_trajectories(
        trajs=pos_trajs_iters[-1], start_state=start_state, goal_state=goal_state,
        plot_trajs=False,
        video_filepath=f'{base_file_name}-robot-traj.mp4',
        n_frames=pos_trajs_iters[-1].shape[1],
        anim_time=n_support_points*dt
    )

    plt.show()


