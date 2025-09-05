import torch

from mp_baselines.planners.rrt_connect import RRTConnect


class DualRobotTaskAdapter:
    """
    适配两个单机器人 task 到联合配置空间，用于双机器人 RRT 规划。

    要求每个单机器人 task 提供以下接口（与 RRTBase 兼容）：
    - random_coll_free_q(n_samples, max_samples=1000)
    - random_q()
    - compute_collision(qs) -> bool 或 bool 向量（碰撞为 True）
    - 距离度量可选；若未提供，将使用拼接空间上的欧氏距离

    inter_collision_fn: callable(q1, q2) -> bool 或 bool 向量
        机器人间相互碰撞检测（必须由上层提供，否则默认不检测）。
        支持批量输入：q1 [N, d1] / q2 [N, d2] 或单个 [d1]/[d2]。
    """

    def __init__(self, task1, task2, q1_dim=None, q2_dim=None, inter_collision_fn=None, tensor_args=None):
        self.task1 = task1
        self.task2 = task2
        self.tensor_args = tensor_args if tensor_args is not None else getattr(task1, 'tensor_args', None)

        # 维度：优先显式传入；否则从 random_q 推断
        if q1_dim is None or q2_dim is None:
            q1_example = task1.random_q()
            q2_example = task2.random_q()
            q1_dim = q1_example.shape[-1]
            q2_dim = q2_example.shape[-1]
        self.q1_dim = int(q1_dim)
        self.q2_dim = int(q2_dim)

        # 机器人间碰撞函数；默认无相互碰撞
        self.inter_collision_fn = inter_collision_fn if inter_collision_fn is not None else self._no_inter_collision

    def _no_inter_collision(self, q1, q2):
        # 始终返回 False（无碰撞），支持标量或批量
        if isinstance(q1, torch.Tensor):
            return torch.zeros(q1.shape[:-1], dtype=torch.bool, device=q1.device)
        return False

    def _split(self, q):
        if q.ndim == 1:
            return q[:self.q1_dim], q[self.q1_dim:self.q1_dim + self.q2_dim]
        else:
            return q[..., :self.q1_dim], q[..., self.q1_dim:self.q1_dim + self.q2_dim]

    def random_q(self):
        q1 = self.task1.random_q()
        q2 = self.task2.random_q()
        return torch.cat([q1, q2], dim=-1)

    def random_coll_free_q(self, n_samples, max_samples=1000, **observation):
        # 逐次拒绝采样，确保各自无碰撞且相互不碰撞
        samples = []
        trials = 0
        device = self.tensor_args['device'] if isinstance(self.tensor_args, dict) and 'device' in self.tensor_args else None
        while len(samples) < n_samples and trials < max_samples:
            trials += 1
            q1 = self.task1.random_coll_free_q(1, max_samples=1, **observation)
            q2 = self.task2.random_coll_free_q(1, max_samples=1, **observation)
            if q1 is None or q2 is None:
                continue
            if q1.ndim > 1:
                q1 = q1[0]
            if q2.ndim > 1:
                q2 = q2[0]
            inter = self.inter_collision_fn(q1.unsqueeze(0), q2.unsqueeze(0))
            inter = inter.squeeze().item() if isinstance(inter, torch.Tensor) else bool(inter)
            if inter:
                continue
            q = torch.cat([q1, q2], dim=-1)
            if device is not None and q.device != torch.device(device):
                q = q.to(device)
            samples.append(q)

        if len(samples) == 0:
            return None
        return torch.stack(samples, dim=0)

    def compute_collision(self, qs, **observation):
        # 输入可为 [D] 或 [N, D]
        single = False
        if qs.ndim == 1:
            qs = qs.unsqueeze(0)
            single = True
        q1, q2 = self._split(qs)
        coll1 = self.task1.compute_collision(q1, **observation)
        coll2 = self.task2.compute_collision(q2, **observation)
        inter = self.inter_collision_fn(q1, q2)
        if not isinstance(coll1, torch.Tensor):
            coll1 = torch.tensor(coll1, dtype=torch.bool, device=qs.device)
        if not isinstance(coll2, torch.Tensor):
            coll2 = torch.tensor(coll2, dtype=torch.bool, device=qs.device)
        if not isinstance(inter, torch.Tensor):
            inter = torch.tensor(inter, dtype=torch.bool, device=qs.device)
        coll = (coll1.bool() | coll2.bool() | inter.bool()).reshape(-1)
        return coll[0] if single else coll

    def distance_q(self, q1, q2):
        # 兼容批量：[N, D] vs [D]
        diff = q1 - q2
        if diff.ndim == 1:
            return torch.linalg.norm(diff, ord=2)
        else:
            return torch.linalg.norm(diff, ord=2, dim=-1)


class RRTConnectDual(RRTConnect):
    """
    双机器人 RRT-Connect：在联合配置空间中进行采样、扩展与连接。

    参数：
        task1, task2: 单机器人任务实例（需与 RRTBase 接口兼容）
        start_state_pos_1, goal_state_pos_1: 机器人1的起点/终点配置
        start_state_pos_2, goal_state_pos_2: 机器人2的起点/终点配置
        inter_collision_fn: 机器人间碰撞检测函数，可批量
    返回：
        optimize() 输出两条轨迹 (traj1, traj2)，形状均为 [T, d_i]；若失败返回 None
    """

    def __init__(
            self,
            task1,
            task2,
            n_iters: int = None,
            start_state_pos_1: torch.Tensor = None,
            goal_state_pos_1: torch.Tensor = None,
            start_state_pos_2: torch.Tensor = None,
            goal_state_pos_2: torch.Tensor = None,
            inter_collision_fn=None,
            step_size: float = 0.1,
            n_radius: float = 1.,
            max_time: float = 60.,
            tensor_args: dict = None,
            n_pre_samples=10000,
            pre_samples=None,
            **kwargs
    ):
        assert start_state_pos_1 is not None and goal_state_pos_1 is not None
        assert start_state_pos_2 is not None and goal_state_pos_2 is not None

        self.adapter = DualRobotTaskAdapter(
            task1,
            task2,
            q1_dim=start_state_pos_1.shape[-1],
            q2_dim=start_state_pos_2.shape[-1],
            inter_collision_fn=inter_collision_fn,
            tensor_args=tensor_args,
        )

        start_concat = torch.cat([start_state_pos_1, start_state_pos_2], dim=-1)
        goal_concat = torch.cat([goal_state_pos_1, goal_state_pos_2], dim=-1)

        super(RRTConnectDual, self).__init__(
            task=self.adapter,
            n_iters=n_iters,
            start_state_pos=start_concat,
            step_size=step_size,
            n_radius=n_radius,
            max_time=max_time,
            goal_state_pos=goal_concat,
            tensor_args=tensor_args,
            n_pre_samples=n_pre_samples,
            pre_samples=pre_samples,
            **kwargs
        )

        self.q1_dim = self.adapter.q1_dim
        self.q2_dim = self.adapter.q2_dim

    def optimize(self, opt_iters=None, **observation):
        path_concat = super(RRTConnectDual, self).optimize(opt_iters=opt_iters, **observation)
        if path_concat is None:
            return None
        # 统一 tensor 形式
        if isinstance(path_concat, list):
            path_concat = torch.stack(path_concat, dim=0)
        traj1 = path_concat[..., :self.q1_dim]
        traj2 = path_concat[..., self.q1_dim:self.q1_dim + self.q2_dim]
        return traj1, traj2

    def render(self, ax, **kwargs):
        # 使用父类渲染两棵树（在联合空间中），如需分别渲染两机器人，可在上层按需拆分
        return super(RRTConnectDual, self).render(ax, **kwargs)


