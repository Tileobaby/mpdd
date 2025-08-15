from isaacgym import gymapi
gym = gymapi.acquire_gym()
sim = gym.create_sim(0,0,gymapi.SIM_PHYSX,gymapi.SimParams())
opts = gymapi.AssetOptions(); opts.fix_base_link=True; opts.disable_gravity=True
asset = gym.load_asset(sim,
    "/home/tian/mpd-public/deps/isaacgym/assets/urdf/fanuc/urdf",
    "fanuc.urdf",
    opts)
assert asset is not None, "URDF failed to load"
print("DOFs:", gym.get_asset_dof_count(asset))      # expect 6
print("Bodies:", gym.get_asset_rigid_body_count(asset))
from isaacgym import gymapi
gym = gymapi.acquire_gym()
sim = gym.create_sim(0,0,gymapi.SIM_PHYSX,gymapi.SimParams())
opts = gymapi.AssetOptions(); opts.fix_base_link=True; opts.disable_gravity=True
asset = gym.load_asset(sim,
    "/home/tian/mpd-public/deps/isaacgym/assets/urdf/fanuc/urdf",
    "fanuc.urdf",
    opts)
assert asset is not None, "URDF failed to load"
print("DOFs:", gym.get_asset_dof_count(asset))      # expect 6
print("Bodies:", gym.get_asset_rigid_body_count(asset))
