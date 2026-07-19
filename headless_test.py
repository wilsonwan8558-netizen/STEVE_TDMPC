import numpy as np
import eve

vessel_tree = eve.intervention.vesseltree.AorticArch(
    seed=30,
    scaling_xyzd=[1.0, 1.0, 1.0, 0.75],
)

device = eve.intervention.device.JShaped()

simulation = eve.intervention.simulation.SofaBeamAdapter(
    friction=0.001
)

fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
    simulation=simulation,
    vessel_tree=vessel_tree,
    image_frequency=7.5,
    image_rot_zx=[20, 5],
)

target = eve.intervention.target.CenterlineRandom(
    vessel_tree=vessel_tree,
    fluoroscopy=fluoroscopy,
    threshold=5,
    branches=["lcca", "rcca", "lsa", "rsa", "bct", "co"],
)

intervention = eve.intervention.MonoPlaneStatic(
    vessel_tree=vessel_tree,
    devices=[device],
    simulation=simulation,
    fluoroscopy=fluoroscopy,
    target=target,
)

start = eve.start.MaxDeviceLength(
    intervention=intervention,
    max_length=500,
)

pathfinder = eve.pathfinder.BruteForceBFS(
    intervention=intervention
)

position = eve.observation.Tracking2D(
    intervention=intervention,
    n_points=5,
)
position = eve.observation.wrapper.NormalizeTracking2DEpisode(
    position,
    intervention,
)

target_state = eve.observation.Target2D(
    intervention=intervention
)
target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
    target_state,
    intervention,
)

rotation = eve.observation.Rotations(
    intervention=intervention
)

observation = eve.observation.ObsDict(
    {
        "position": position,
        "target": target_state,
        "rotation": rotation,
    }
)

reward = eve.reward.Combination(
    [
        eve.reward.TargetReached(
            intervention=intervention,
            factor=1.0,
        ),
        eve.reward.PathLengthDelta(
            pathfinder=pathfinder,
            factor=0.01,
        ),
    ]
)

terminal = eve.terminal.TargetReached(
    intervention=intervention
)

truncation = eve.truncation.MaxSteps(200)

env = eve.Env(
    intervention=intervention,
    observation=observation,
    reward=reward,
    terminal=terminal,
    truncation=truncation,
    start=start,
    pathfinder=pathfinder,
)

obs, info = env.reset()
print("Reset observation:", obs)

for step in range(20):
    action = (35.0, 1.0)

    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"step={step}, reward={reward:.4f}, "
        f"terminated={terminated}, truncated={truncated}"
    )

    if terminated or truncated:
        break

env.close()
print("Headless stEVE test passed.")