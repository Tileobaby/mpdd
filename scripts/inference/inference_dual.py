# inference.py  (Revised for TWO Fanuc robots with alternating diffusion updates)
# NOTE: Your original single-robot sampling block is preserved but COMMENTED below.

from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController
from torch_robotics.isaac_gym_envs.motion_planning_envs_fanuc import FanucMotionPlanningIsaacGymEnv

import os
import pickle
from math import ceil
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from experiment_launcher import single_experiment_yaml, run_experiment
# --- Costs (use the module that exists in your repo) ---
# If your project uses mp_baselines.planners.costs.costs, keep that import and comment the other.
# from mp_baselines.planners.costs.costs import CostCollision, CostComposite, CostGPTrajectory
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory
# New inter-robot cost (you just added it)
from mp_baselines.planners.costs.cost_functions import CostInterRobotCollision

from mpd.models import TemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mpd.models.diffusion_models.sample_functions import ddpm_sample_fn, guide_gradient_steps

from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml

from torch_robotics.robots import RobotPanda, RobotFanuc
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

allow_ops_in_compiled_graph()

TRAINED_MODELS_DIR = '../../data_trained_models/'

# ============================================================================================
# Helper: Alternating (A -> B) diffusion sampler.
# Requires: GaussianDiffusionModel.p_sample_step(...) method (added to your diffusion class).
# If you moved this helper to mpd.models.diffusion_models.sample_functions, you can import it
# and delete this local definition.

import math

def make_T(x=0.0, y=0.0, z=0.0, yaw=0.0, device="cuda", dtype=torch.float32):
    c, s = math.cos(yaw), math.sin(yaw)
    T = torch.tensor([[ c, -s, 0., x],
                      [ s,  c, 0., y],
                      [0., 0., 1., z],
                      [0., 0., 0., 1.]], device=device, dtype=dtype)
    return T

def apply_T_to_positions(pos_bhn3: torch.Tensor, T_44: torch.Tensor) -> torch.Tensor:
    """
    pos: (B,H,N,3) or (BH,N,3) ; T: (4,4)
    returns same shape with world offset applied.
    """
    if pos_bhn3.dim() == 3:   # (BH, N, 3)
        ph = torch.cat([pos_bhn3, torch.ones_like(pos_bhn3[..., :1])], dim=-1)      # (BH,N,4)
        pw = torch.einsum('ij,bnj->bni', T_44, ph)[..., :3]                          # (BH,N,3)
        return pw
    elif pos_bhn3.dim() == 4: # (B, H, N, 3)
        ph = torch.cat([pos_bhn3, torch.ones_like(pos_bhn3[..., :1])], dim=-1)      # (B,H,N,4)
        pw = torch.einsum('ij,bhnj->bhni', T_44, ph)[..., :3]                        # (B,H,N,3)
        return pw
    else:
        raise RuntimeError(f"Unexpected position tensor shape: {tuple(pos_bhn3.shape)}")


@torch.no_grad()
def alternating_block_gibbs_sample(
    model_a,
    model_b,
    *,
    horizon_a, horizon_b,
    state_dim_a, state_dim_b,
    hard_conds_a, hard_conds_b,     # dicts, already repeated to batch = n_samples
    context_a=None, context_b=None,
    return_chain=True,
    n_diffusion_steps_without_noise=0,
    sample_kwargs_a=None,
    sample_kwargs_b=None,
):
    sample_kwargs_a = sample_kwargs_a or {}
    sample_kwargs_b = sample_kwargs_b or {}

    device = model_a.betas.device
    any_key = next(iter(hard_conds_a))
    B = hard_conds_a[any_key].shape[0]

    x_a = torch.randn(B, horizon_a, state_dim_a, device=device)
    x_b = torch.randn(B, horizon_b, state_dim_b, device=device)
    # Apply hard conditions
    from mpd.models.diffusion_models.sample_functions import apply_hard_conditioning
    x_a = apply_hard_conditioning(x_a, hard_conds_a)
    x_b = apply_hard_conditioning(x_b, hard_conds_b)

    chain_a = [x_a] if return_chain else None
    chain_b = [x_b] if return_chain else None

    # Optional: if guide exposes set_other_traj, pass the other trajectory each step
    def set_other_if_supported(kw, other):
        g = kw.get('guide', None)
        if g is not None and hasattr(g, 'set_other_traj'):
            g.set_other_traj(other)

    for t in reversed(range(-n_diffusion_steps_without_noise, model_a.n_diffusion_steps)):
        # A step (B frozen)
        set_other_if_supported(sample_kwargs_a, x_b)
        x_a = model_a.p_sample_step(x_a, hard_conds_a, context=context_a, t=t, **sample_kwargs_a)
        if return_chain:
            chain_a.append(x_a)

        # B step (use freshly updated A)
        set_other_if_supported(sample_kwargs_b, x_a)
        x_b = model_b.p_sample_step(x_b, hard_conds_b, context=context_b, t=t, **sample_kwargs_b)
        if return_chain:
            chain_b.append(x_b)

    if return_chain:
        chain_a = torch.stack(chain_a, dim=1)  # (B,S,H,D)
        chain_b = torch.stack(chain_b, dim=1)
        chain_a = einops.rearrange(chain_a, 'b s h d -> s b h d')
        chain_b = einops.rearrange(chain_b, 'b s h d -> s b h d')
        return chain_a, chain_b

    return x_a, x_b
# ============================================================================================


@single_experiment_yaml
def experiment(
    ########################################################################################################################
    # Experiment configuration
    model_id: str = 'EnvSpheres3D-RobotFanuc',
    #model_id: str = 'EnvSpheres3D-RobotPanda',  # NEW: Fanuc dual-robot model

    # planner_alg: str = 'diffusion_prior',
    # planner_alg: str = 'diffusion_prior_then_guide',
    planner_alg: str = 'mpd',   # online guidance during diffusion

    use_guide_on_extra_objects_only: bool = False,

    n_samples: int = 50,

    start_guide_steps_fraction: float = 0.25,
    n_guide_steps: int = 5,
    n_diffusion_steps_without_noise: int = 5,

    weight_grad_cost_collision: float = 1e-2,
    weight_grad_cost_smoothness: float = 1e-7,
    weight_grad_cost_inter_robot: float = 5e-3,  # NEW: weight for inter-robot cost

    factor_num_interpolated_points_for_collision: float = 1.5,

    trajectory_duration: float = 5.0,  # currently fixed

    ########################################################################
    device: str = 'cuda',

    debug: bool = True,

    render: bool = True,

    ########################################################################
    # MANDATORY
    seed: int = 30,
    results_dir: str = 'logs',

    ########################################################################
    **kwargs
):
    ########################################################################################################################
    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    ########################################################################################################################
    print(f'##########################################################################################################')
    print(f'Model -- {model_id}')
    print(f'Algorithm -- {planner_alg}')
    run_prior_only = False
    run_prior_then_guidance = False
    if planner_alg == 'mpd':
        pass
    elif planner_alg == 'diffusion_prior_then_guide':
        run_prior_then_guidance = True
    elif planner_alg == 'diffusion_prior':
        run_prior_only = True
    else:
        raise NotImplementedError

    ########################################################################################################################
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    results_dir = os.path.join(model_dir, 'results_inference_dual_fanuc', str(seed))  # NEW output dir
    os.makedirs(results_dir, exist_ok=True)

    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

    ########################################################################################################################
    # Load dataset with env, robot, task (Fanuc)
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        use_extra_objects=True,
        obstacle_cutoff_margin=0.05,
        **args,
        tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    n_support_points = dataset.n_support_points
    env = dataset.env
    robot = dataset.robot          # Fanuc (we’ll treat as Robot A)
    task = dataset.task

    # Use the same Fanuc model for Robot B (identical geometry, separate motion)
    # keep dataset.robot as the reference geometry/limits
    robot_a = RobotFanuc(use_collision_spheres=robot.use_collision_spheres, tensor_args=tensor_args)
    robot_b = RobotFanuc(use_collision_spheres=robot.use_collision_spheres, tensor_args=tensor_args)

    # copy joint limits if needed (usually RobotFanuc sets them internally already)
    robot_a.q_min, robot_a.q_max = robot.q_min, robot.q_max
    robot_b.q_min, robot_b.q_max = robot.q_min, robot.q_max

    dt = trajectory_duration / n_support_points
    
    robot_a.dt = dt
    robot_b.dt = dt

    device = tensor_args['device']
    T_world_base_a = make_T(0.0, 0.0, 0.0, yaw=0.0, device=device)   # keep A at origin
    T_world_base_b = make_T(0.8, 0.0, 0.0, yaw=0.0, device=device)   # move B +0.8 m in X

    # keep bound originals
    _fk_a = robot_a.fk_map_collision
    _fk_b = robot_b.fk_map_collision

    def fk_map_collision_a(q_pos):
        return _fk_a(q_pos)  # unchanged

    def fk_map_collision_b(q_pos):
        pos = _fk_b(q_pos)                 # (B,H,N,3) or (BH,N,3)
        return apply_T_to_positions(pos, T_world_base_b)

    robot_a.fk_map_collision = fk_map_collision_a
    robot_b.fk_map_collision = fk_map_collision_b

    # --- also offset Robot B's collision spheres so inter-robot cost sees the shift ---
    if hasattr(robot_b, "get_collision_spheres"):
        _get_spheres_b = robot_b.get_collision_spheres
        def get_collision_spheres_b(trajs):
            pos, rad = _get_spheres_b(trajs)        # pos: (B,H,N,3) or (BH,N,3)
            pos = apply_T_to_positions(pos, T_world_base_b)
            return pos, rad
        robot_b.get_collision_spheres = get_collision_spheres_b


    #dt = trajectory_duration / n_support_points
    #robot_a.dt = dt
    #robot_b.dt = dt

    ########################################################################################################################
    # Load prior model (reuse for both Fanucs)
    diffusion_configs = dict(
        variance_schedule=args['variance_schedule'],
        n_diffusion_steps=args['n_diffusion_steps'],
        predict_epsilon=args['predict_epsilon'],
    )
    unet_configs = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=args['unet_input_dim'],
        dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']],
    )
    diffusion_model = get_model(
        model_class=args['diffusion_model_class'],
        model=TemporalUnet(**unet_configs),
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs
    )
    diffusion_model.load_state_dict(
        torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'),
        map_location=tensor_args['device'])
    )
    diffusion_model.eval()

    # We use the SAME prior for A & B (Fanuc + Fanuc)
    model_a = diffusion_model
    model_b = diffusion_model

    freeze_torch_model_params(model_a)
    freeze_torch_model_params(model_b)
    model_a = torch.compile(model_a)
    model_b = torch.compile(model_b)
    model_a.warmup(horizon=n_support_points, device=device)
    model_b.warmup(horizon=n_support_points, device=device)

    ########################################################################################################################
    # Random initial and final positions for BOTH robots (env-only collision free)
    def sample_start_goal():
        n_tries = 100
        for _ in range(n_tries):
            q_free = task.random_coll_free_q(n_samples=2)
            start, goal = q_free[0], q_free[1]
            if torch.linalg.norm(start - goal) > dataset.threshold_start_goal_pos:
                return start, goal
        raise ValueError("No collision-free start/goal found.")

    start_state_pos_a, goal_state_pos_a = sample_start_goal()
    start_state_pos_b, goal_state_pos_b = sample_start_goal()

    print(f'start_state_pos_a: {start_state_pos_a}')
    print(f'goal_state_pos_a:  {goal_state_pos_a}')
    print(f'start_state_pos_b: {start_state_pos_b}')
    print(f'goal_state_pos_b:  {goal_state_pos_b}')

    ########################################################################################################################
    # HARD conditions (normalized) for both robots
    hard_conds_a = dataset.get_hard_conditions(torch.vstack((start_state_pos_a, goal_state_pos_a)), normalize=True)
    hard_conds_b = dataset.get_hard_conditions(torch.vstack((start_state_pos_b, goal_state_pos_b)), normalize=True)

    context = None  # no extra context model here
    t_start_guide = ceil(start_guide_steps_fraction * model_a.n_diffusion_steps)

    ########################################################################################################################
    # COSTS & GUIDES (env collision + smoothness + inter-robot)
    if use_guide_on_extra_objects_only:
        collision_fields = task.get_collision_fields_extra_objects()
    else:
        collision_fields = task.get_collision_fields()

    def build_env_collision_costs(robot_ref):
        out = []
        for cf in collision_fields:
            out.append(
                CostCollision(
                    robot_ref, n_support_points,
                    field=cf,
                    sigma_coll=1.0,
                    tensor_args=tensor_args
                )
            )
        return out

    cost_collision_a = build_env_collision_costs(robot_a)
    cost_collision_b = build_env_collision_costs(robot_b)

    cost_smooth_a = [CostGPTrajectory(robot_a, n_support_points, dt, sigma_gp=1.0, tensor_args=tensor_args)]
    cost_smooth_b = [CostGPTrajectory(robot_b, n_support_points, dt, sigma_gp=1.0, tensor_args=tensor_args)]

    # NEW: inter-robot collision
    inter_robot_cost_a = [CostInterRobotCollision(robot_a, robot_b, n_support_points, margin=0.02, power=2.0, tensor_args=tensor_args)]
    inter_robot_cost_b = [CostInterRobotCollision(robot_b, robot_a, n_support_points, margin=0.02, power=2.0, tensor_args=tensor_args)]

    weights_a = [weight_grad_cost_collision] * len(cost_collision_a) + [weight_grad_cost_smoothness] * len(cost_smooth_a) + [weight_grad_cost_inter_robot] * len(inter_robot_cost_a)
    weights_b = [weight_grad_cost_collision] * len(cost_collision_b) + [weight_grad_cost_smoothness] * len(cost_smooth_b) + [weight_grad_cost_inter_robot] * len(inter_robot_cost_b)

    cost_composite_a = CostComposite(robot_a, n_support_points,
                                     [*cost_collision_a, *cost_smooth_a, *inter_robot_cost_a],
                                     weights_cost_l=weights_a, tensor_args=tensor_args)
    cost_composite_b = CostComposite(robot_b, n_support_points,
                                     [*cost_collision_b, *cost_smooth_b, *inter_robot_cost_b],
                                     weights_cost_l=weights_b, tensor_args=tensor_args)

    # Let the guide optionally broadcast "other traj" into any cost that supports it

    def _attach_set_other_to_guide(guide_obj, cost_composite, dataset):
        """
        Adds guide_obj.set_other_traj(x_other_norm: (B,H,D) normalized)
        Internally unnormalizes and forwards to any cost that implements set_other_traj(...).
        """
        def set_other_traj(x_other_norm: torch.Tensor):
            # x_other_norm comes from the sampler (normalized). Unnormalize to joint space for costs.
            if x_other_norm.dim() == 3:  # (B,H,D) -> make a fake "chain" so API matches
                chain = x_other_norm.unsqueeze(0)        # (1,B,H,D)
                x_other_unnorm = dataset.unnormalize_trajectories(chain)[-1]  # (B,H,D)
            else:
                # Fallback if already shaped like a chain
                x_other_unnorm = dataset.unnormalize_trajectories(x_other_norm)[-1]

            for c in cost_composite.cost_l:
                if hasattr(c, "set_other_traj"):
                    c.set_other_traj(x_other_unnorm)

        # monkey-patch the method onto the guide
        guide_obj.set_other_traj = set_other_traj

    guide_a = GuideManagerTrajectoriesWithVelocity(
        dataset,
        cost_composite_a,
        clip_grad=True,
        interpolate_trajectories_for_collision=True,
        num_interpolated_points=ceil(n_support_points * factor_num_interpolated_points_for_collision),
        tensor_args=tensor_args,
    )
    guide_b = GuideManagerTrajectoriesWithVelocity(
        dataset,
        cost_composite_b,
        clip_grad=True,
        interpolate_trajectories_for_collision=True,
        num_interpolated_points=ceil(n_support_points * factor_num_interpolated_points_for_collision),
        tensor_args=tensor_args,
    )
    _attach_set_other_to_guide(guide_a, cost_composite_a, dataset)
    _attach_set_other_to_guide(guide_b, cost_composite_b, dataset)

    ########################################################################################################################
    # ========================== ORIGINAL SINGLE-ROBOT SAMPLING (COMMENTED) ==========================
    # with TimerCUDA() as timer_model_sampling:
    #     trajs_normalized_iters = model_a.run_inference(
    #         context, hard_conds_a,
    #         n_samples=n_samples, horizon=n_support_points,
    #         return_chain=True,
    #         sample_fn=ddpm_sample_fn,
    #         guide=None if run_prior_then_guidance or run_prior_only else guide_a,
    #         n_guide_steps=n_guide_steps,
    #         t_start_guide=t_start_guide,
    #         noise_std_extra_schedule_fn=lambda x: 0.5,
    #         n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
    #     )
    # print(f'[SINGLE] t_model_sampling: {timer_model_sampling.elapsed:.3f} sec')
    # t_total = timer_model_sampling.elapsed
    # ===============================================================================================

    ########################################################################################################################
    # ========================== NEW: DUAL-ROBOT ALTERNATING DIFFUSION SAMPLING =====================
    sample_kwargs_a = dict(
        guide=None if run_prior_then_guidance or run_prior_only else guide_a,
        n_guide_steps=n_guide_steps,
        t_start_guide=t_start_guide,
        noise_std_extra_schedule_fn=lambda x: 0.5,
    )
    sample_kwargs_b = dict(
        guide=None if run_prior_then_guidance or run_prior_only else guide_b,
        n_guide_steps=n_guide_steps,
        t_start_guide=t_start_guide,
        noise_std_extra_schedule_fn=lambda x: 0.5,
    )

    hard_conds_a_rep = {k: einops.repeat(v, 'd -> b d', b=n_samples) for k, v in hard_conds_a.items()}
    hard_conds_b_rep = {k: einops.repeat(v, 'd -> b d', b=n_samples) for k, v in hard_conds_b.items()}

    with TimerCUDA() as timer_model_sampling:
        trajs_chain_a_norm, trajs_chain_b_norm = alternating_block_gibbs_sample(
            model_a=model_a,
            model_b=model_b,
            horizon_a=n_support_points, horizon_b=n_support_points,
            state_dim_a=dataset.state_dim, state_dim_b=dataset.state_dim,
            hard_conds_a=hard_conds_a_rep,
            hard_conds_b=hard_conds_b_rep,
            context_a=None, context_b=None,
            return_chain=True,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
            sample_kwargs_a=sample_kwargs_a,
            sample_kwargs_b=sample_kwargs_b,
        )
    print(f'[DUAL] t_model_sampling: {timer_model_sampling.elapsed:.3f} sec')
    t_total = timer_model_sampling.elapsed
    # ===============================================================================================

    ######## Optional: Post-diffusion guide-only clean-up (PRIOR-THEN-GUIDE mode)
    if run_prior_then_guidance:
        n_post = (t_start_guide + n_diffusion_steps_without_noise) * n_guide_steps
        with TimerCUDA() as timer_post:
            trajs_a = trajs_chain_a_norm[-1]
            trajs_b = trajs_chain_b_norm[-1]
            post_a, post_b = [], []
            for _ in range(n_post):
                guide_a.set_other_traj(trajs_b)
                guide_b.set_other_traj(trajs_a)
                trajs_a = guide_gradient_steps(trajs_a, hard_conds=hard_conds_a_rep, guide=guide_a, n_guide_steps=1, unnormalize_data=False)
                trajs_b = guide_gradient_steps(trajs_b, hard_conds=hard_conds_b_rep, guide=guide_b, n_guide_steps=1, unnormalize_data=False)
                post_a.append(trajs_a); post_b.append(trajs_b)
            chain_a = torch.stack(post_a, dim=1)
            chain_b = torch.stack(post_b, dim=1)
            chain_a = einops.rearrange(chain_a, 'b p h d -> p b h d')
            chain_b = einops.rearrange(chain_b, 'b p h d -> p b h d')
            trajs_chain_a_norm = torch.cat((trajs_chain_a_norm, chain_a))
            trajs_chain_b_norm = torch.cat((trajs_chain_b_norm, chain_b))
        print(f'[DUAL] t_post_diffusion_guide: {timer_post.elapsed:.3f} sec')
        t_total = timer_model_sampling.elapsed + timer_post.elapsed

    ########################################################################################################################
    # Unnormalize to joint space
    trajs_chain_a = dataset.unnormalize_trajectories(trajs_chain_a_norm)  # (S, B, H, D)
    trajs_chain_b = dataset.unnormalize_trajectories(trajs_chain_b_norm)
    trajs_final_a = trajs_chain_a[-1]   # (B, H, D)
    trajs_final_b = trajs_chain_b[-1]

    # Per-robot env collision split (inter-robot handled in metrics/success)
    trajs_final_a_coll, idxs_a_coll, trajs_final_a_free, idxs_a_free, _ = task.get_trajs_collision_and_free(trajs_final_a, return_indices=True)
    trajs_final_b_coll, idxs_b_coll, trajs_final_b_free, idxs_b_free, _ = task.get_trajs_collision_and_free(trajs_final_b, return_indices=True)

    ########################################################################################################################
    # Metrics
    print(f'\n----------------METRICS----------------')
    print(f't_total: {t_total:.3f} sec')

    success_free_trajs_a = task.compute_success_free_trajs(trajs_final_a)
    success_free_trajs_b = task.compute_success_free_trajs(trajs_final_b)
    fraction_free_trajs_a = task.compute_fraction_free_trajs(trajs_final_a)
    fraction_free_trajs_b = task.compute_fraction_free_trajs(trajs_final_b)
    collision_intensity_a = task.compute_collision_intensity_trajs(trajs_final_a)
    collision_intensity_b = task.compute_collision_intensity_trajs(trajs_final_b)
    print(f'success A: {success_free_trajs_a} | success B: {success_free_trajs_b}')
    print(f'free % A: {fraction_free_trajs_a*100:.2f} | free % B: {fraction_free_trajs_b*100:.2f}')
    print(f'collision intensity A: {collision_intensity_a*100:.2f} | B: {collision_intensity_b*100:.2f}')

    # TODO: add inter-robot collision metrics here (min pairwise link distance across time)

    traj_final_free_best_a, traj_final_free_best_b = None, None
    idx_best_a, idx_best_b = None, None

    if trajs_final_a_free is not None and trajs_final_a_free.shape[0] > 0:
        smooth_a = compute_smoothness(trajs_final_a_free, robot_a)
        path_a = compute_path_length(trajs_final_a_free, robot_a)
        cost_all_a = smooth_a + path_a
        idx_best_a = torch.argmin(cost_all_a).item()
        traj_final_free_best_a = trajs_final_a_free[idx_best_a]
        print(f'A smooth mean/std: {smooth_a.mean():.4f}/{smooth_a.std():.4f} | path mean/std: {path_a.mean():.4f}/{path_a.std():.4f}')
    if trajs_final_b_free is not None and trajs_final_b_free.shape[0] > 0:
        smooth_b = compute_smoothness(trajs_final_b_free, robot_b)
        path_b = compute_path_length(trajs_final_b_free, robot_b)
        cost_all_b = smooth_b + path_b
        idx_best_b = torch.argmin(cost_all_b).item()
        traj_final_free_best_b = trajs_final_b_free[idx_best_b]
        print(f'B smooth mean/std: {smooth_b.mean():.4f}/{smooth_b.std():.4f} | path mean/std: {path_b.mean():.4f}/{path_b.std():.4f}')

    print(f'\n--------------------------------------\n')

    ########################################################################################################################
    # Save data
    results_data_dict = {
        'trajs_chain_a': trajs_chain_a,
        'trajs_chain_b': trajs_chain_b,
        'trajs_final_a_coll': trajs_final_a_coll,
        'trajs_final_a_free': trajs_final_a_free,
        'trajs_final_b_coll': trajs_final_b_coll,
        'trajs_final_b_free': trajs_final_b_free,
        'idx_best_a': idx_best_a,
        'idx_best_b': idx_best_b,
        'traj_final_free_best_a': traj_final_free_best_a,
        'traj_final_free_best_b': traj_final_free_best_b,
        't_total': t_total,
    }
    with open(os.path.join(results_dir, 'results_data_dict_dual_fanuc.pickle'), 'wb') as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ########################################################################################################################
    # Render results (single-robot visualizations kept; dual-actor Isaac render TODO)
    if render:
        planner_visualizer = PlanningVisualizer(task=task)
        base_file_name = Path(os.path.basename(__file__)).stem

        # ================== ORIGINAL SINGLE-ROBOT ISAAC RENDER (COMMENTED) ==================
        # if isinstance(robot_a, RobotFanuc) and traj_final_free_best_a is not None:
        #     n_first_steps = 10; n_last_steps = 10
        #     trajs_pos_a = robot_a.get_position(trajs_final_a_free).movedim(1, 0)
        #     trajs_pos_a = interpolate_traj_via_points(trajs_pos_a.movedim(0, 1), 2).movedim(1, 0)
        #     motion_planning_isaac_env = FanucMotionPlanningIsaacGymEnv(
        #         env, robot_a, task,
        #         asset_root="../../deps/isaacgym/assets",
        #         fanuc_asset_file="urdf/fanuc/urdf/fanuc.urdf",
        #         ee_link_name="tool0",
        #         controller_type='position',
        #         num_envs=trajs_pos_a.shape[1],
        #         all_robots_in_one_env=True,
        #         color_robots=False,
        #         show_goal_configuration=True,
        #         sync_with_real_time=True,
        #         show_collision_spheres=False,
        #         dt=dt,
        #     )
        #     motion_planning_controller = MotionPlanningController(motion_planning_isaac_env)
        #     motion_planning_controller.run_trajectories(
        #         trajs_pos_a,
        #         start_states_joint_pos=trajs_pos_a[0], goal_state_joint_pos=trajs_pos_a[-1][0],
        #         n_first_steps=n_first_steps,
        #         n_last_steps=n_last_steps,
        #         visualize=True,
        #         render_viewer_camera=True,
        #         make_video=True,
        #         video_path=os.path.join(results_dir, f'{base_file_name}-isaac-controller-position-A.mp4'),
        #         make_gif=False
        #     )
        # ====================================================================================
                # --- DUAL FANUC ISAAC RENDER (two robots in one env) ---
        # --- DUAL FANUC ISAAC RENDER (two robots in one env) ---
        if isinstance(robot_a, RobotFanuc) and isinstance(robot_b, RobotFanuc):
            # choose a paired sample to visualize (prefer one that's env-free for both robots)
            idx_vis = 0
            try:
                if (idxs_a_free is not None) and (idxs_b_free is not None):
                    common = set(idxs_a_free.tolist()) & set(idxs_b_free.tolist())
                    if len(common) > 0:
                        idx_vis = sorted(list(common))[0]
            except Exception:
                pass

            # joints over time for A and B, shape: (H, D)
            q_traj_a = robot_a.get_position(trajs_final_a[idx_vis:idx_vis+1]).squeeze(0)   # (H, D)
            q_traj_b = robot_b.get_position(trajs_final_b[idx_vis:idx_vis+1]).squeeze(0)   # (H, D)

            # stack into (H, 2, D) then densify -> (T, 2, D)
            trajs_pos_pair = torch.stack([q_traj_a, q_traj_b], dim=1)  # (H, 2, D)
            _tr = trajs_pos_pair.movedim(0, 1).contiguous()            # (2, H, D)
            _tr = interpolate_traj_via_points(_tr, 2).contiguous()     # (2, T, D)
            trajs_pos_pair = _tr.movedim(1, 0).contiguous()            # (T, 2, D)

            n_first_steps = 10
            n_last_steps = 10

            # Build an Isaac env with TWO Fanucs sharing one viewer
            motion_planning_isaac_env = FanucMotionPlanningIsaacGymEnv(
                env, robot_a, task,
                asset_root="../../deps/isaacgym/assets",
                fanuc_asset_file="urdf/fanuc/urdf/fanuc.urdf",
                ee_link_name="tool0",
                controller_type='position',
                num_envs=trajs_pos_pair.shape[1],   # 2 robots
                all_robots_in_one_env=True,         # same scene
                color_robots=True,                  # easier to tell apart
                show_goal_configuration=False,      # <-- disable single-goal ghost
                sync_with_real_time=True,
                show_collision_spheres=False,
                dt=dt,
            )
            #test
            num_envs = len(motion_planning_isaac_env.envs)
            envh = motion_planning_isaac_env.envs[0]
            print("num_envs:", num_envs, "actors in env[0]:", motion_planning_isaac_env.gym.get_actor_count(envh))
            for i in range(motion_planning_isaac_env.gym.get_actor_count(envh)):
                ah = motion_planning_isaac_env.gym.get_actor_handle(envh, i)
                print(i, motion_planning_isaac_env.gym.get_actor_name(envh, ah))
            #debug

            from isaacgym import gymapi, gymtorch

            gym  = motion_planning_isaac_env.gym
            sim  = motion_planning_isaac_env.sim
            envh = motion_planning_isaac_env.envs[0]   # single env when all_robots_in_one_env=True

            # Refresh first, then wrap the current root-state tensor
            gym.refresh_actor_root_state_tensor(sim)
            root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))

            # Find the two Fanuc actors in this env
            num_actors = gym.get_actor_count(envh)
            robot_handles = []
            for i in range(num_actors):
                ah   = gym.get_actor_handle(envh, i)
                name = gym.get_actor_name(envh, ah)
                if name and "fanuc" in name.lower():
                    robot_handles.append(ah)

            # Fallback: if names aren’t informative, just take the last two actors
            if len(robot_handles) < 2:
                robot_handles = [gym.get_actor_handle(envh, num_actors - 2),
                                gym.get_actor_handle(envh, num_actors - 1)]

            # Map second robot handle → SIM root-state index
            actor_b_handle  = robot_handles[1]
            actor_b_sim_idx = gym.get_actor_index(envh, actor_b_handle, gymapi.DOMAIN_SIM)

            # Shift robot B’s base by +0.8 m in X (must match your planner’s offset)
            root[actor_b_sim_idx, 0:3] = torch.tensor([0.8, 0.0, 0.0], device=root.device, dtype=root.dtype)

            # Push the updated root states back to the simulator
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))


            motion_planning_controller = MotionPlanningController(motion_planning_isaac_env)

            # Pass a single (D,) goal vector to satisfy API (won't draw a goal ghost)
            goal_vec = trajs_pos_pair[-1, 0]  # (D,) e.g., robot A's final
            motion_planning_controller.run_trajectories(
                trajs_pos_pair,                           # (T, 2, D)
                start_states_joint_pos=trajs_pos_pair[0], # (2, D)
                goal_state_joint_pos=goal_vec,            # (D,)  <-- single vector
                n_first_steps=n_first_steps,
                n_last_steps=n_last_steps,
                visualize=True,
                render_viewer_camera=True,
                make_video=True,
                video_path=os.path.join(results_dir, f'{base_file_name}-dual-isaac-controller-position.mp4'),
                make_gif=False
            )



        # Planning visualizer (per-robot)
        if traj_final_free_best_a is not None:
            planner_visualizer.animate_opt_iters_joint_space_state(
                trajs=trajs_chain_a,
                pos_start_state=start_state_pos_a, pos_goal_state=goal_state_pos_a,
                vel_start_state=torch.zeros_like(start_state_pos_a), vel_goal_state=torch.zeros_like(goal_state_pos_a),
                traj_best=traj_final_free_best_a,
                video_filepath=os.path.join(results_dir, f'{base_file_name}-A-joint-space-opt-iters.mp4'),
                n_frames=max((2, len(trajs_chain_a))),
                anim_time=5
            )
        if traj_final_free_best_b is not None:
            planner_visualizer.animate_opt_iters_joint_space_state(
                trajs=trajs_chain_b,
                pos_start_state=start_state_pos_b, pos_goal_state=goal_state_pos_b,
                vel_start_state=torch.zeros_like(start_state_pos_b), vel_goal_state=torch.zeros_like(goal_state_pos_b),
                traj_best=traj_final_free_best_b,
                video_filepath=os.path.join(results_dir, f'{base_file_name}-B-joint-space-opt-iters.mp4'),
                n_frames=max((2, len(trajs_chain_b))),
                anim_time=5
            )

        plt.show()


if __name__ == '__main__':
    run_experiment(experiment)
