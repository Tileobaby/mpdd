import os
import numpy as np
import torch
import cv2
from math import ceil

from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_numpy
from torch_robotics.torch_kinematics_tree.models.robots import modidy_franka_panda_urdf_grasped_object  # not used here

# If you already have these helpers, you can remove these copies.
def set_position_and_orientation(center, obj_pos, obj_ori):
    obj_pose = gymapi.Transform()
    obj_pose.p = gymapi.Vec3(*(center + obj_pos))
    obj_pose.r = gymapi.Quat(obj_ori[1], obj_ori[2], obj_ori[3], obj_ori[0])
    return obj_pose

def create_assets_from_primitive_shapes(sim, gym, obj_list):
    from torch_robotics.environments.primitives import MultiSphereField, MultiBoxField
    from torch_robotics.torch_utils.torch_utils import to_numpy
    object_assets_l, object_poses_l = [], []
    for obj in obj_list or []:
        obj_pos = to_numpy(obj.pos)
        obj_ori = to_numpy(obj.ori)
        for obj_field in obj.fields:
            if isinstance(obj_field, MultiSphereField):
                for center, radius in zip(obj_field.centers, obj_field.radii):
                    center_np = to_numpy(center); radius_np = to_numpy(radius)
                    asset_options = gymapi.AssetOptions(); asset_options.fix_base_link = True
                    sphere_asset = gym.create_sphere(sim, float(radius_np), asset_options)
                    object_assets_l.append(sphere_asset)
                    object_poses_l.append(set_position_and_orientation(center_np, obj_pos, obj_ori))
            elif isinstance(obj_field, MultiBoxField):
                for center, size in zip(obj_field.centers, obj_field.sizes):
                    center_np = to_numpy(center); size_np = to_numpy(size)
                    asset_options = gymapi.AssetOptions(); asset_options.fix_base_link = True
                    box_asset = gym.create_box(sim, float(size_np[0]), float(size_np[1]), float(size_np[2]), asset_options)
                    object_assets_l.append(box_asset)
                    object_poses_l.append(set_position_and_orientation(center_np, obj_pos, obj_ori))
            else:
                raise NotImplementedError
    return object_assets_l, object_poses_l

class ViewerRecorder:
    def __init__(self, dt=1/50., fps=50):
        self.dt = dt; self.fps = fps; self.step_img = []
    def append(self, step, img): self.step_img.append((step, img))
    def make_video(self, video_path='./trajs_replay.mp4', n_first_steps=0, n_last_steps=0, make_gif=False):
        from moviepy.video.io.VideoFileClip import VideoFileClip
        if not self.step_img: return
        frame0 = self.step_img[0][1]; h,w,_ = frame0.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, self.fps, (w,h))
        max_steps = len(self.step_img) - 1
        image_l = []
        for step, frame in self.step_img:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if step > n_first_steps:
                step_text = 0 if step > len(self.step_img) - n_last_steps - 1 else step - n_first_steps
                cv2.putText(frame, f'Step: {step_text}/{max_steps-n_first_steps-n_last_steps}', (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_4)
                cv2.putText(frame, f'Time: {self.dt*step_text:.2f} secs', (50,85),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_4)
            video.write(frame); image_l.append(frame)
        cv2.destroyAllWindows(); video.release()
        if make_gif:
            clip = VideoFileClip(video_path)
            gif_path = os.path.splitext(video_path)[0] + '.gif'
            clip.write_gif(gif_path, fps=self.fps)

class FanucMotionPlanningIsaacGymEnv:
    """
    Isaac Gym environment tailored for a 6-DOF Fanuc arm (no gripper).
    DOF counts and tensor shapes are inferred from the loaded asset, so
    you can swap to other 6-DOF Fanucs without code edits.
    """
    def __init__(
        self, env, robot, task,
        asset_root="../../deps/isaacgym/assets",
        fanuc_asset_file="urdf/fanuc/robots/fanuc_m10ia.urdf",  # <-- put your URDF here
        ee_link_name="tool0",  # <-- set to your URDF's EE link
        controller_type='position',
        num_envs=8,
        all_robots_in_one_env=False,
        color_robots=False,
        use_pipeline_gpu=False,
        show_goal_configuration=True,
        sync_with_real_time=False,
        show_collision_spheres=False,
        color_robots_in_collision=False,
        show_contact_forces=False,
        dt=1./25.,
        lower_level_controller_frequency=1000,
        **kwargs,
    ):
        self.env = env
        self.robot = robot
        self.task = task
        self.controller_type = controller_type
        self.num_envs = num_envs + 1 if show_goal_configuration else num_envs

        self.all_robots_in_one_env = all_robots_in_one_env
        self.color_robots = color_robots
        self.color_robots_in_collision = color_robots_in_collision
        self.show_collision_spheres = show_collision_spheres
        self.show_contact_forces = show_contact_forces

        self.ee_link_name = ee_link_name
        self.sync_with_real_time = sync_with_real_time

        # Gym setup
        self.gym = gymapi.acquire_gym()
        self.gym_args = gymutil.parse_arguments()
        self.gym_args.use_gpu_pipeline = use_pipeline_gpu
        self.tensor_args = {'device': get_torch_device(device='cuda' if use_pipeline_gpu else 'cpu')}

        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = dt
        sim_params.substeps = ceil(lower_level_controller_frequency * dt)
        sim_params.use_gpu_pipeline = self.gym_args.use_gpu_pipeline
        if self.gym_args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = self.gym_args.num_threads
            sim_params.physx.use_gpu = self.gym_args.use_gpu
        else:
            raise Exception("Use PhysX for this example")

        self.sim = self.gym.create_sim(
            self.gym_args.compute_device_id, self.gym_args.graphics_device_id, self.gym_args.physics_engine, sim_params)
        if self.sim is None: raise Exception("Failed to create sim")

        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None: raise Exception("Failed to create viewer")

        # Environment assets
        obj_fixed_assets, obj_fixed_poses = create_assets_from_primitive_shapes(self.sim, self.gym, self.env.obj_fixed_list)
        obj_extra_assets, obj_extra_poses = create_assets_from_primitive_shapes(self.sim, self.gym, self.env.obj_extra_list)

        # Load Fanuc asset
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        arm_asset = self.gym.load_asset(self.sim, asset_root, fanuc_asset_file, asset_options)

        # DOF info (Fanuc is 6 DOF; no gripper)
        self.total_dofs = self.gym.get_asset_dof_count(arm_asset)  # expect 6
        self.n_arm_dofs = self.total_dofs

        arm_dof_props = self.gym.get_asset_dof_properties(arm_asset)
        self.arm_lower_limits = arm_dof_props["lower"]
        self.arm_upper_limits = arm_dof_props["upper"]

        # Configure drive
        if self.controller_type == 'position':
            arm_dof_props["driveMode"][:self.n_arm_dofs].fill(gymapi.DOF_MODE_POS)
            arm_dof_props["stiffness"][:self.n_arm_dofs].fill(400.0)
            arm_dof_props["damping"][:self.n_arm_dofs] = 2.0 * np.sqrt(arm_dof_props["stiffness"][:self.n_arm_dofs])
        elif self.controller_type == 'velocity':
            arm_dof_props["driveMode"][:self.n_arm_dofs].fill(gymapi.DOF_MODE_VEL)
            arm_dof_props["stiffness"][:self.n_arm_dofs].fill(0.0)
            arm_dof_props["damping"][:self.n_arm_dofs].fill(600.0)
        else:
            raise NotImplementedError

        # Defaults
        self.default_dof_pos = np.zeros(self.total_dofs, dtype=np.float32)
        self.default_dof_state = np.zeros(self.total_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = self.default_dof_pos

        # Grid of envs
        num_per_row = int(np.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # Ground
        plane_params = gymapi.PlaneParams()
        plane_params.distance = 2
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # Robot base pose
        arm_pose = gymapi.Transform()
        arm_pose.p = gymapi.Vec3(0, 0, 0)

        self.envs = []
        self.arm_handles = []
        self.obj_idxs = []
        self.hand_idxs = []
        self.map_rigid_body_idxs_to_env_idx = {}
        self.show_goal_configuration = show_goal_configuration
        self.goal_joint_position = None

        color_obj_fixed = gymapi.Vec3(220./255., 220./255., 220./255.)
        color_obj_extra = gymapi.Vec3(1., 0., 0.)

        if self.all_robots_in_one_env:
            env0 = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env0)

        # Create envs + actors
        for i in range(self.num_envs):
            if not self.all_robots_in_one_env:
                env_i = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
                self.envs.append(env_i)
            else:
                env_i = self.envs[0]

            # Fixed objs
            for obj_asset, obj_pose in zip(obj_fixed_assets, obj_fixed_poses):
                obj_h = self.gym.create_actor(env_i, obj_asset, obj_pose, "obj_fixed", i, 0)
                self.gym.set_rigid_body_color(env_i, obj_h, 0, gymapi.MESH_VISUAL_AND_COLLISION, color_obj_fixed)
                idx = self.gym.get_actor_rigid_body_index(env_i, obj_h, 0, gymapi.DOMAIN_SIM)
                self.obj_idxs.append(idx); self.map_rigid_body_idxs_to_env_idx[idx] = i

            # Extra objs
            for obj_asset, obj_pose in zip(obj_extra_assets, obj_extra_poses):
                obj_h = self.gym.create_actor(env_i, obj_asset, obj_pose, "obj_extra", i, 0)
                self.gym.set_rigid_body_color(env_i, obj_h, 0, gymapi.MESH_VISUAL_AND_COLLISION, color_obj_extra)
                idx = self.gym.get_actor_rigid_body_index(env_i, obj_h, 0, gymapi.DOMAIN_SIM)
                self.obj_idxs.append(idx); self.map_rigid_body_idxs_to_env_idx[idx] = i

            # Fanuc actor
            arm_h = self.gym.create_actor(env_i, arm_asset, arm_pose, "fanuc", i, 0)
            self.arm_handles.append(arm_h)

            # Color goal env purple
            n_rb = self.gym.get_actor_rigid_body_count(env_i, arm_h)
            if self.show_goal_configuration and i == self.num_envs - 1:
                color = gymapi.Vec3(128/255., 0., 128/255.)
                for j in range(n_rb):
                    self.gym.set_rigid_body_color(env_i, arm_h, j, gymapi.MESH_VISUAL_AND_COLLISION, color)

            if color_robots and not (self.show_goal_configuration and i == self.num_envs - 1):
                c = np.random.random(3)
                color = gymapi.Vec3(c[0], c[1], c[2])
                for j in range(n_rb):
                    self.gym.set_rigid_body_color(env_i, arm_h, j, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # DOF props per actor
            self.gym.set_actor_dof_properties(env_i, arm_h, arm_dof_props)

        # Camera pointing to middle env
        cam_pos = gymapi.Vec3(0, 1.75, 1.25)
        cam_target = gymapi.Vec3(0, -3, -1.25)
        self.middle_env = self.envs[0] if len(self.envs) == 1 else self.envs[self.num_envs // 2 + num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, self.middle_env, cam_pos, cam_target)

        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280; camera_props.height = 720
        self.viewer_camera_handle = self.gym.create_camera_sensor(self.middle_env, camera_props)
        self.gym.set_camera_location(self.viewer_camera_handle, self.middle_env, cam_pos, cam_target)

        self.viewer_recorder = ViewerRecorder(dt=sim_params.dt, fps=ceil(1 / sim_params.dt))

        # Debug geoms
        self.axes_geom = gymutil.AxesGeometry(0.15)

        # Prepare tensor API
        self.gym.prepare_sim(self.sim)

        # Tensors
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        # Shape: (num_envs, total_dofs, 1)
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, self.total_dofs, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, self.total_dofs, 1)

        self.step_idx = 0

    def reset(self, start_joint_positions=None, goal_joint_position=None):
        self.step_idx = 0

        if start_joint_positions is None:
            start_joint_positions = torch.zeros((self.num_envs, self.n_arm_dofs), **self.tensor_args)

        start_joint_positions = start_joint_positions.to(**self.tensor_args)
        assert start_joint_positions.ndim == 2 and start_joint_positions.shape[1] == self.n_arm_dofs

        dof_pos_tensor = torch.zeros((self.num_envs, self.total_dofs), **self.tensor_args)

        if self.show_goal_configuration:
            dof_pos_tensor[:-1, :self.n_arm_dofs] = start_joint_positions
            assert goal_joint_position is not None
            self.goal_joint_position = goal_joint_position.to(**self.tensor_args)
            dof_pos_tensor[-1, :self.n_arm_dofs] = self.goal_joint_position
        else:
            dof_pos_tensor[:, :self.n_arm_dofs] = start_joint_positions

        # Apply actor DOF states/targets actor-by-actor (keeps parity with your Panda env)
        envs = (self.envs * self.num_envs) if self.all_robots_in_one_env else self.envs
        for env_i, handle, joints_pos in zip(envs, self.arm_handles, dof_pos_tensor):
            joint_state_des = self.gym.get_actor_dof_states(env_i, handle, gymapi.STATE_ALL)
            joint_state_des['pos'] = np.zeros_like(joint_state_des['pos'])
            joint_state_des['vel'] = np.zeros_like(joint_state_des['vel'])
            jp_np = to_numpy(joints_pos[:self.n_arm_dofs])
            joint_state_des['pos'][:self.n_arm_dofs] = jp_np
            self.gym.set_actor_dof_states(env_i, handle, joint_state_des, gymapi.STATE_ALL)
            self.gym.set_actor_dof_position_targets(env_i, handle, joint_state_des['pos'])

        # Refresh, return current joint states
        self.gym.refresh_dof_state_tensor(self.sim)
        joint_states_curr = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(self.num_envs, self.total_dofs, 2)
        if self.show_goal_configuration:
            joint_states_curr = joint_states_curr[:-1, ...]
        return joint_states_curr

    def step(self, actions, visualize=True, render_viewer_camera=False):
        # Physics step
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Refresh
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # Build action tensor (num_envs, total_dofs)
        action_dof = torch.zeros_like(self.dof_pos).squeeze(-1)
        if self.show_goal_configuration:
            action_dof[:-1, :self.n_arm_dofs] = actions[..., :self.n_arm_dofs]
            if self.controller_type == 'position':
                action_dof[-1, :self.n_arm_dofs] = self.goal_joint_position
            elif self.controller_type == 'velocity':
                action_dof[-1, :self.n_arm_dofs] = 0.0
            else:
                raise NotImplementedError
        else:
            action_dof[..., :self.n_arm_dofs] = actions[..., :self.n_arm_dofs]

        # Send to sim
        if self.controller_type == 'position':
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action_dof))
        elif self.controller_type == 'velocity':
            self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(action_dof))

        # Detect contacts
        envs = (self.envs * self.num_envs) if self.all_robots_in_one_env else self.envs
        arm_handles = self.arm_handles
        if self.show_goal_configuration:
            envs = envs[:-1]; arm_handles = arm_handles[:-1]

        envs_with_robot_in_contact = []
        for env_i, _ in zip(envs, arm_handles):
            rigid_contacts = self.gym.get_env_rigid_contacts(env_i)
            if self.all_robots_in_one_env:
                for contact in rigid_contacts:
                    body1_idx = contact[2]
                    env_idx = self.map_rigid_body_idxs_to_env_idx.get(body1_idx, None)
                    if env_idx is not None and env_idx not in envs_with_robot_in_contact:
                        envs_with_robot_in_contact.append(env_idx)
            else:
                if len(rigid_contacts) > 0:
                    env_idx = rigid_contacts[0][0]
                    if env_idx not in envs_with_robot_in_contact:
                        envs_with_robot_in_contact.append(env_idx)

        # Viz
        if visualize:
            self.gym.clear_lines(self.viewer)
            envs_draw = (self.envs * self.num_envs) if self.all_robots_in_one_env else self.envs
            for k, (env_i, arm_h) in enumerate(zip(envs_draw, self.arm_handles)):
                # EE frame
                body_dict = self.gym.get_actor_rigid_body_dict(env_i, arm_h)
                props = self.gym.get_actor_rigid_body_states(env_i, arm_h, gymapi.STATE_POS)
                # robust EE link lookup
                ee_index = body_dict.get(self.ee_link_name, None)
                if ee_index is None:
                    # fallback: last link
                    ee_index = len(props['pose']) - 1
                ee_pose = props['pose'][:][ee_index]
                ee_tf = gymapi.Transform(p=gymapi.Vec3(*ee_pose[0]), r=gymapi.Quat(*ee_pose[1]))
                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env_i, ee_tf)

                # Collision coloring
                if self.color_robots_in_collision and k in envs_with_robot_in_contact:
                    n_rb = self.gym.get_actor_rigid_body_count(env_i, arm_h)
                    color = gymapi.Vec3(0., 0., 0.)
                    for j in range(n_rb):
                        self.gym.set_rigid_body_color(env_i, arm_h, j, gymapi.MESH_VISUAL_AND_COLLISION, color)

                # (Optional) show contact vectors
                if self.show_contact_forces:
                    self.gym.draw_env_rigid_contacts(self.viewer, env_i, gymapi.Vec3(1,0,0), 0.5, True)

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            if self.sync_with_real_time:
                self.gym.sync_frame_time(self.sim)

            if render_viewer_camera:
                self.gym.render_all_camera_sensors(self.sim)
                viewer_img = self.gym.get_camera_image(self.sim, self.middle_env, self.viewer_camera_handle, gymapi.IMAGE_COLOR)
                viewer_img = viewer_img.reshape(viewer_img.shape[0], -1, 4)[..., :3]
                self.viewer_recorder.append(self.step_idx, viewer_img)

        self.step_idx += 1

        joint_states_curr = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(self.num_envs, self.total_dofs, 2)
        if self.show_goal_configuration:
            joint_states_curr = joint_states_curr[:-1, ...]
        return joint_states_curr, envs_with_robot_in_contact

    def check_viewer_has_closed(self):
        return self.gym.query_viewer_has_closed(self.viewer)

    def clean_up(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
