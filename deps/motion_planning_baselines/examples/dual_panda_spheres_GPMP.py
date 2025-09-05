import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mp_baselines.planners.gpmp2 import GPMP2
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
    跨机器人碰撞代价：使用联合机器人链接点（B,H,T,3）按两台 Panda 拆分，
    计算两两最小距离，并对小于 margin 的部分施加惩罚。
    """

    def __init__(self, robot_composite: CompositePandaRobot, margin: float = 0.04, tensor_args=None):
        self.robot_composite = robot_composite
        self.margin = margin
        self.tensor_args = tensor_args if tensor_args is not None else robot_composite.tensor_args

        self._n_links_r1 = len(robot_composite.robot1.diff_panda.get_link_names())
        self._n_links_r2 = len(robot_composite.robot2.diff_panda.get_link_names())

    def compute_cost(self, q, link_pos, **kwargs):
        # 统一形状为 (B,H,T,3)
        if link_pos.ndim == 3:  # (B, T, 3)
            link_pos = link_pos.unsqueeze(1)
        elif link_pos.ndim != 4:
            raise NotImplementedError

        B, H, T, D = link_pos.shape
        assert D == 3
        assert T >= (self._n_links_r1 + self._n_links_r2)

        pos_r1 = link_pos[..., :self._n_links_r1, :]
        pos_r2 = link_pos[..., self._n_links_r1:self._n_links_r1 + self._n_links_r2, :]

        # pairwise distances (B,H,L1,L2)
        diff = pos_r1.unsqueeze(-2) - pos_r2.unsqueeze(-3)
        dists = torch.linalg.norm(diff, dim=-1)

        min_over_l2 = torch.min(dists, dim=-1)[0]      # (B,H,L1)
        min_dist = torch.min(min_over_l2, dim=-1)[0]   # (B,H)

        penalty = torch.relu(self.margin - min_dist)
        return penalty

    def zero_grad(self):
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
    inter_field = InterRobotCollisionField(composite_robot, margin=0.04, tensor_args=tensor_args)

    def sample_dual_free_q(max_trials=1000):
        for _ in range(max_trials):
            q_free = task.random_coll_free_q(n_samples=2)
            start_state = q_free[0]
            goal_state = q_free[1]
            # inter-robot check for start/goal
            lp = composite_robot.fk_map_collision(start_state.unsqueeze(0).unsqueeze(0))
            if inter_field.compute_cost(None, lp).max() > 0:
                continue
            lp = composite_robot.fk_map_collision(goal_state.unsqueeze(0).unsqueeze(0))
            if inter_field.compute_cost(None, lp).max() > 0:
                continue
            return start_state, goal_state
        raise RuntimeError("Failed to sample dual-robot non-inter-colliding start/goal")

    start_state, goal_state = sample_dual_free_q()

    print(start_state)
    print(goal_state)

    # -------------------------------- Planner params (manual, since env params expect single robot) ----------------
    duration = 5  # sec
    n_support_points = 128
    dt = duration / n_support_points

    num_particles_per_goal = 10

    # sigma/defaults adapted from EnvSpheres3D.get_gpmp2_params for single Panda
    gpmp_params = dict(
        opt_iters=120,
        num_samples=64,
        sigma_start=1e-3,
        sigma_gp=1e-1,
        sigma_coll=1e-4,
        sigma_goal_prior=1e-3,
        step_size=1.0,
        sigma_start_init=1e-4,
        sigma_goal_init=1e-4,
        sigma_gp_init=0.1,
        sigma_start_sample=1e-3,
        sigma_goal_sample=1e-3,
        solver_params={
            'delta': 1e-2,
            'trust_region': True,
            'sparse_computation': False,
            'sparse_computation_block_diag': False,
            'method': 'cholesky',
        },
        stop_criteria=0.1,
    )

    # collision fields: env/self/boundary from task + inter-robot
    collision_fields = [*task.get_collision_fields(), inter_field]

    planner_params = dict(
        robot=composite_robot,
        n_dof=composite_robot.q_dim,
        n_support_points=n_support_points,
        num_particles_per_goal=num_particles_per_goal,
        opt_iters=gpmp_params['opt_iters'],
        dt=dt,
        start_state=start_state,
        multi_goal_states=goal_state.unsqueeze(0),
        sigma_start_init=gpmp_params['sigma_start_init'],
        sigma_goal_init=gpmp_params['sigma_goal_init'],
        sigma_gp_init=gpmp_params['sigma_gp_init'],
        sigma_start_sample=gpmp_params['sigma_start_sample'],
        sigma_goal_sample=gpmp_params['sigma_goal_sample'],
        solver_params=gpmp_params['solver_params'],
        stop_criteria=gpmp_params['stop_criteria'],
        # forwarded to build_gpmp2_cost_composite
        sigma_start=gpmp_params['sigma_start'],
        sigma_gp=gpmp_params['sigma_gp'],
        sigma_coll=gpmp_params['sigma_coll'],
        sigma_goal_prior=gpmp_params['sigma_goal_prior'],
        num_samples=gpmp_params['num_samples'],
        collision_fields=collision_fields,
        tensor_args=tensor_args,
    )

    planner = GPMP2(**planner_params)

    # -------------------------------- Optimize ---------------------------------
    opt_iters = gpmp_params['opt_iters']
    trajs_0 = planner.get_traj()
    trajs_iters = []
    trajs_iters.append(trajs_0)
    costs_previous = None
    with TimerCUDA() as t:
        for i in range(opt_iters):
            print(f'Iteration: {i}')
            trajs = planner.optimize(opt_iters=1, debug=True)
            trajs_iters.append(trajs)

            costs = planner.costs
            if i == 0:
                costs_previous = costs
                continue

            if torch.all(torch.abs((costs - costs_previous)/costs) < 0.1):
                break

            costs_previous = costs.clone()

    trajs_iters = torch.stack(trajs_iters)
    print(f'Optimization time: {t.elapsed:.3f} sec')

    # -------------------------------- Visualize & Save ---------------------------------
    planner_visualizer = PlanningVisualizer(
        task=task,
        planner=planner
    )

    print(f'----------------STATISTICS----------------')
    print(f'percentage free trajs: {task.compute_fraction_free_trajs(trajs_iters[-1])*100:.2f}')
    print(f'percentage collision intensity {task.compute_collision_intensity_trajs(trajs_iters[-1])*100:.2f}')
    print(f'success {task.compute_success_free_trajs(trajs_iters[-1])}')

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
        n_frames=max((2, len(trajs_iters) // 10)),
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

    # 保存结果
    results_data_dict = {
        'duration': duration,
        'n_support_points': n_support_points,
        'dt': dt,
        'trajs_iters_all': trajs_iters.detach().cpu(),
    }
    with open(os.path.join('./', f'{base_file_name}-results_data_dict.pickle'), 'wb') as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.show()


