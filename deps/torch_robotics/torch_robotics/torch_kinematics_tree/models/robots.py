from pathlib import Path
from typing import Optional, List
from xml.dom import minidom

import numpy as np
import yaml
from urdf_parser_py.urdf import URDF, Joint, Link, Visual, Collision, Box, Pose

from torch_robotics.torch_kinematics_tree.geometrics.quaternion import q_to_euler
from torch_robotics.torch_kinematics_tree.models.robot_tree import DifferentiableTree
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path, get_configs_path
from xml.etree import ElementTree as ET

from torch_robotics.torch_utils.torch_utils import to_numpy


class DifferentiableKUKAiiwa(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'kuka_iiwa' / 'urdf' / 'iiwa7.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_kuka_iiwa"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


def modidy_franka_panda_urdf_grasped_object(robot_file, grasped_object):
    robot_urdf = URDF.from_xml_file(robot_file)
    joint = Joint(
        name='grasped_object_fixed_joint',
        parent='panda_hand',
        child='grasped_object',
        joint_type='fixed',
        origin=Pose(xyz=to_numpy(grasped_object.pos.squeeze()),
                    rpy=to_numpy(q_to_euler(grasped_object.ori).squeeze())
                    )
    )
    robot_urdf.add_joint(joint)

    geometry_grasped_object = grasped_object.geometry_urdf
    link = Link(
        name='grasped_object',
        visual=Visual(geometry_grasped_object),
        # inertial=None,
        collision=Collision(geometry_grasped_object),
        origin=Pose(xyz=[0., 0., 0.], rpy=[0., 0., 0.])
    )
    robot_urdf.add_link(link)

    # replace the robots file
    robot_file = Path(str(robot_file).replace('.urdf', '_grasped_object.urdf'))
    xmlstr = minidom.parseString(ET.tostring(robot_urdf.to_xml())).toprettyxml(indent="   ")
    with open(str(robot_file), "w") as f:
        f.write(xmlstr)

    return robot_file


def modidy_franka_panda_urdf_collision_model(robot_file):
    collision_spheres = 'panda/panda_sphere_config.yaml'
    # load collision file:
    coll_yml = (get_configs_path() / collision_spheres).as_posix()
    with open(coll_yml) as file:
        coll_params = yaml.load(file, Loader=yaml.FullLoader)

    robot_urdf = URDF.from_xml_file(robot_file)

    link_collision_names = []
    link_collision_margins = []
    for link_name, spheres_l in coll_params.items():
        for i, sphere in enumerate(spheres_l):
            joint = Joint(
                name=f'joint_{link_name}_sphere_{i}',
                parent=f'{link_name}',
                child=f'{link_name}_{i}',
                joint_type='fixed',
                origin=Pose(xyz=to_numpy(sphere[:3]))
            )
            robot_urdf.add_joint(joint)

            link_collision = f'{link_name}_{i}'
            link = Link(
                name=link_collision,
                origin=Pose(xyz=[0., 0., 0.], rpy=[0., 0., 0.])
            )
            robot_urdf.add_link(link)

            link_collision_names.append(link_collision)
            link_collision_margins.append(sphere[-1])

    # replace the robots file
    robot_file = Path(str(robot_file).replace('.urdf', '_collision_model.urdf'))
    xmlstr = minidom.parseString(ET.tostring(robot_urdf.to_xml())).toprettyxml(indent="   ")
    with open(str(robot_file), "w") as f:
        f.write(xmlstr)

    return robot_file, link_collision_names, link_collision_margins


class DifferentiableFrankaPanda(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, gripper=False, device='cpu', grasped_object=None,
                 use_collision_spheres=False):
        if gripper:
            robot_file = get_robot_path() / 'franka_description' / 'robots' / 'panda_arm_hand.urdf'
        else:
            robot_file = get_robot_path() / 'franka_description' / 'robots' / 'panda_arm_no_gripper.urdf'

        # Modify the urdf to append links of the collision model
        if use_collision_spheres:
            robot_file, link_collision_names, link_collision_margins = modidy_franka_panda_urdf_collision_model(robot_file)
            self.link_collision_names = link_collision_names
            self.link_collision_margins = link_collision_margins

        # Modify the urdf to append the link of the grasped object
        if grasped_object is not None:
            robot_file = modidy_franka_panda_urdf_grasped_object(robot_file, grasped_object)

        self.model_path = robot_file.as_posix()
        self.name = "differentiable_franka_panda"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableUR10(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, attach_gripper=False, device='cpu'):
        robot_path = get_robot_path()
        if attach_gripper:
            robot_file = robot_path / 'ur10' / 'urdf' / 'ur10_suction.urdf'
        else:
            robot_file = robot_path / 'ur10' / 'urdf' / 'ur10.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_ur10"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableHabitatStretch(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_path = get_robot_path()
        robot_file = robot_path / 'habitat_stretch' / 'urdf' / 'hab_stretch.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_stretch"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableTiagoDualHolo(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'tiago_dual_description' / 'tiago_dual_holobase_minimal.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_tiago_dual_holo"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableTiagoDualHoloMove(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'tiago_dual_description' / 'tiago_dual_holobase_minimal_holonomic.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_tiago_dual_holo_move"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)

    def get_link_names(self):  # pop those hacky frames for moving base
        return super().get_link_names()[3:]


class DifferentiableShadowHand(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'shadow_hand' / 'shadow_hand.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_shadow_hand"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableAllegroHand(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'allegro_hand' / 'allegro_hand.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_allegro_hand"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class Differentiable2LinkPlanar(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'planar_manipulators' / 'urdf' / '2_link_planar.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_2_link_planar"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


#add
class DifferentiableFanuc(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu', use_collision_spheres=False):
        robot_file = get_robot_path() / 'fanuc' / 'urdf' / 'fanuc.urdf'
        
        # Add collision spheres support if requested
        if use_collision_spheres:
            robot_file, link_collision_names, link_collision_margins = self.modify_fanuc_urdf_collision_model(robot_file)
            self.link_collision_names = link_collision_names
            self.link_collision_margins = link_collision_margins
        
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_fanuc"
        self.ee_link = "tool0"  # <- Set EE link here
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)

    def get_joint_limit_array(self):
        jl_lower = [-2.9671, -1.9199, -1.2043, -3.3161, -2.0944, -6.2832]
        jl_upper = [ 2.9671,  2.0944,  3.5779,  3.3161,  2.0944,  6.2832]
        jl_vel   = [ 8.03,    8.03,    9.08,    9.77,    9.77,   15.71]
        jl_eff   = [150,      150,     150,     150,     150,    150]
        return jl_lower, jl_upper, jl_vel, jl_eff
    
    def modify_fanuc_urdf_collision_model(self, robot_file):
        """Add collision spheres to Fanuc URDF - optimized for LRMate200iD geometry"""
        
        # Define collision spheres for Fanuc LRMate200iD robot links
        # Based on the URDF joint positions and typical Fanuc geometry
        fanuc_collision_spheres = {
            'base_link': [
                [0.0, 0.0, 0.165, 0.15],   # Base cylinder, centered at mid-height
                [0.0, 0.0, 0.08, 0.12],    # Lower base section
            ],
            'link_1': [
                [0.0, 0.0, 0.0, 0.12],     # Joint 1 housing
                [0.0, 0.0, -0.1, 0.1],     # Lower connection
            ],
            'link_2': [
                [0.0, 0.0, 0.0, 0.11],     # Joint 2 housing
                [0.0, 0.0, 0.13, 0.09],    # Mid-section towards joint 3
                [0.0, 0.0, 0.22, 0.08],    # Near joint 3 connection
            ],
            'link_3': [
                [0.0, 0.0, 0.0, 0.08],     # Joint 3 base
                [0.0, 0.0, 0.01, 0.07],    # Small offset for link geometry
            ],
            'link_4': [
                [0.0, 0.0, 0.0, 0.06],     # Joint 4 housing (wrist roll)
                [0.145, 0.0, 0.0, 0.05],   # Mid-forearm (halfway to joint 5)
                [0.25, 0.0, 0.0, 0.045],   # Near joint 5 connection
            ],
            'link_5': [
                [0.0, 0.0, 0.0, 0.045],    # Joint 5 housing (wrist pitch)
                [0.035, 0.0, 0.0, 0.04],   # Mid-section to joint 6
            ],
            'link_6': [
                [0.0, 0.0, 0.0, 0.04],     # Joint 6 housing (wrist yaw)
            ],
            'flange': [
                [0.0, 0.0, 0.0, 0.03],     # Flange connection
            ],
            'tool0': [
                [0.0, 0.0, 0.0, 0.025],    # Tool center point
            ],
        }
        
        robot_urdf = URDF.from_xml_file(robot_file)
        
        link_collision_names = []
        link_collision_margins = []
        
        for link_name, spheres_list in fanuc_collision_spheres.items():
            for i, sphere in enumerate(spheres_list):
                joint = Joint(
                    name=f'joint_{link_name}_sphere_{i}',
                    parent=f'{link_name}',
                    child=f'{link_name}_{i}',
                    joint_type='fixed',
                    origin=Pose(xyz=sphere[:3])  # x, y, z position
                )
                robot_urdf.add_joint(joint)

                link_collision = f'{link_name}_{i}'
                link = Link(
                    name=link_collision,
                    origin=Pose(xyz=[0., 0., 0.], rpy=[0., 0., 0.])
                )
                robot_urdf.add_link(link)

                link_collision_names.append(link_collision)
                link_collision_margins.append(sphere[3])  # radius

        # Save modified URDF
        robot_file = Path(str(robot_file).replace('.urdf', '_collision_model.urdf'))
        xmlstr = minidom.parseString(ET.tostring(robot_urdf.to_xml())).toprettyxml(indent="   ")
        with open(str(robot_file), "w") as f:
            f.write(xmlstr)

        return robot_file, link_collision_names, link_collision_margins