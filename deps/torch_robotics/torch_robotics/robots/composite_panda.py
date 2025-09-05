from typing import Tuple, List

import torch

from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.torch_utils.torch_utils import to_torch


class _CombinedSelfCollisionField:
    """
    Minimal wrapper that aggregates self-collision costs from two robots.
    Matches the interface expected by PlanningTask: compute_cost(q, fk_collision_pos, field_type=..., **kwargs).

    We intentionally recompute FK for each sub-robot to preserve their own
    self-collision configuration (interpolations, margins, etc.).
    """

    def __init__(self, robot1: RobotPanda, robot2: RobotPanda, tensor_args=None):
        self.robot1 = robot1
        self.robot2 = robot2
        self.tensor_args = tensor_args if tensor_args is not None else getattr(robot1, 'tensor_args', None)

    def compute_cost(self, q, fk_collision_pos=None, field_type='sdf', **kwargs):
        if q.ndim == 1:
            q = q.unsqueeze(0)
        q1 = q[..., : self.robot1.q_dim]
        q2 = q[..., self.robot1.q_dim : self.robot1.q_dim + self.robot2.q_dim]

        cost1 = 0
        cost2 = 0
        if self.robot1.df_collision_self is not None:
            fk1 = self.robot1.fk_map_collision(q1)
            cost1 = self.robot1.df_collision_self.compute_cost(q1, fk1, field_type=field_type, **kwargs)
        if self.robot2.df_collision_self is not None:
            fk2 = self.robot2.fk_map_collision(q2)
            cost2 = self.robot2.df_collision_self.compute_cost(q2, fk2, field_type=field_type, **kwargs)

        # costs may be tensors shaped (batch, horizon) or (batch,) depending on field_type
        return cost1 + cost2

    def zero_grad(self):
        # defer to underlying fields if needed
        if self.robot1.df_collision_self is not None and hasattr(self.robot1.df_collision_self, 'zero_grad'):
            self.robot1.df_collision_self.zero_grad()
        if self.robot2.df_collision_self is not None and hasattr(self.robot2.df_collision_self, 'zero_grad'):
            self.robot2.df_collision_self.zero_grad()


class CompositePandaRobot(RobotBase):
    """
    A composite robot that concatenates two Panda robots into a single configuration space.
    - Configuration: q = [q_panda_1, q_panda_2]
    - FK for collision: returns concatenated link positions of both robots
    - Object/Workspace collision checking: uses unified list of link indices and margins
    - Self-collision: wraps two robots' self-collision fields and sums their costs
    """

    def __init__(self,
                 robot1: RobotPanda,
                 robot2: RobotPanda,
                 base_translation_1: torch.Tensor = None,
                 base_translation_2: torch.Tensor = None,
                 tensor_args=None,
                 **kwargs):
        self.robot1 = robot1
        self.robot2 = robot2
        self.tensor_args = tensor_args if tensor_args is not None else getattr(robot1, 'tensor_args', None)

        # base translations (world frame) to separate robots in space
        device = self.tensor_args['device'] if isinstance(self.tensor_args, dict) else 'cpu'
        dtype = self.tensor_args['dtype'] if isinstance(self.tensor_args, dict) else torch.float32
        self.base_translation_1 = base_translation_1 if base_translation_1 is not None else torch.zeros(3, device=device, dtype=dtype)
        self.base_translation_2 = base_translation_2 if base_translation_2 is not None else torch.zeros(3, device=device, dtype=dtype)

        # Compose joint limits and dimension
        q_limits = torch.cat((robot1.q_limits, robot2.q_limits), dim=-1)

        # Build object-collision checking metadata by concatenating per-robot definitions
        link_names_for_object_collision_checking: List[str] = (
            list(robot1.link_names_for_object_collision_checking) +
            list(robot2.link_names_for_object_collision_checking)
        )

        # link margins: concatenate along link dimension
        margins_r1 = robot1.link_margins_for_object_collision_checking
        margins_r2 = robot2.link_margins_for_object_collision_checking
        if isinstance(margins_r1, torch.Tensor):
            margins_r1 = margins_r1
        else:
            margins_r1 = to_torch(margins_r1, **self.tensor_args)
        if isinstance(margins_r2, torch.Tensor):
            margins_r2 = margins_r2
        else:
            margins_r2 = to_torch(margins_r2, **self.tensor_args)
        link_margins_for_object_collision_checking = torch.cat((margins_r1.view(-1, 1), margins_r2.view(-1, 1)), dim=0)

        # Re-index link indices for object collision checking for the second robot with offset
        # RobotPanda.fk_map_collision returns positions for all links in diff model order
        n_links_r1_all = len(self.robot1.diff_panda.get_link_names())
        link_idxs_for_object_collision_checking = (
            list(self.robot1.link_idxs_for_object_collision_checking) +
            [idx + n_links_r1_all for idx in self.robot2.link_idxs_for_object_collision_checking]
        )

        # Interpolation density for object-collision points: sum to keep resolution for both robots
        num_interpolated_points_for_object_collision_checking = (
            int(self.robot1.num_interpolated_points_for_object_collision_checking) +
            int(self.robot2.num_interpolated_points_for_object_collision_checking)
        )

        # Compose self-collision via a small wrapper; do not attempt cross-robot pairs here
        self.df_collision_self = _CombinedSelfCollisionField(self.robot1, self.robot2, tensor_args=self.tensor_args)

        super().__init__(
            name='CompositePandaRobot',
            q_limits=q_limits,
            grasped_object=None,  # handled by individual robots already
            link_names_for_object_collision_checking=link_names_for_object_collision_checking,
            link_margins_for_object_collision_checking=link_margins_for_object_collision_checking,
            link_idxs_for_object_collision_checking=link_idxs_for_object_collision_checking,
            # self-collision in this composite is handled by wrapper above
            link_names_for_self_collision_checking=None,
            link_names_pairs_for_self_collision_checking=None,
            link_idxs_for_self_collision_checking=None,
            num_interpolated_points_for_self_collision_checking=0,
            self_collision_margin_robot=0.0,
            use_collision_spheres=(self.robot1.use_collision_spheres and self.robot2.use_collision_spheres),
            num_interpolated_points_for_object_collision_checking=num_interpolated_points_for_object_collision_checking,
            tensor_args=self.tensor_args,
            **kwargs
        )

    def split_q(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = q[..., : self.robot1.q_dim]
        q2 = q[..., self.robot1.q_dim : self.robot1.q_dim + self.robot2.q_dim]
        return q1, q2

    def fk_map_collision_impl(self, q, **kwargs):
        """
        Return concatenated link positions for collision checking from both robots.
        Expected output shape: (..., num_links_total, 3)
        """
        q_orig_shape = q.shape
        if len(q_orig_shape) == 3:
            b, h, d = q_orig_shape
        elif len(q_orig_shape) == 2:
            b, d = q_orig_shape
            h = 1
        else:
            raise NotImplementedError

        q1, q2 = self.split_q(q)

        fk1 = self.robot1.fk_map_collision(q1, **kwargs)  # (b,h,l1,3)
        fk2 = self.robot2.fk_map_collision(q2, **kwargs)  # (b,h,l2,3)

        # Ensure shapes are (b,h,links,3)
        if len(fk1.shape) == 3:
            fk1 = fk1.unsqueeze(1)
        if len(fk2.shape) == 3:
            fk2 = fk2.unsqueeze(1)

        # apply base translations
        fk1 = fk1 + self.base_translation_1.view(1, 1, 1, 3)
        fk2 = fk2 + self.base_translation_2.view(1, 1, 1, 3)

        link_pos = torch.cat((fk1, fk2), dim=-2)
        return link_pos

    def render(self, ax, q=None, color=('blue', 'green'), draw_links_spheres=False, **kwargs):
        q = self.random_q(1).squeeze() if q is None else q
        q1, q2 = self.split_q(q)
        color1, color2 = (color if isinstance(color, tuple) else ('blue', 'green'))
        self.robot1.render(ax, q=q1, color=color1, draw_links_spheres=draw_links_spheres, **kwargs)
        self.robot2.render(ax, q=q2, color=color2, draw_links_spheres=draw_links_spheres, **kwargs)

    def render_trajectories(self, ax, trajs=None, **kwargs):
        if trajs is None:
            return
        q_pos = self.get_position(trajs)
        q1, q2 = self.split_q(q_pos)
        self.robot1.render_trajectories(ax, trajs=q1, **kwargs)
        self.robot2.render_trajectories(ax, trajs=q2, **kwargs)


