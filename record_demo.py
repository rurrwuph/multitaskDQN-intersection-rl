import glob
import os
import shutil
import subprocess
import random
import numpy as np
import torch

# Flow & TraCI imports
from flow.core.params import (
    VehicleParams,
    NetParams,
    SumoParams,
    EnvParams,
    InitialConfig,
    TrafficLightParams,
    InFlows,
    SumoCarFollowingParams,
)
from flow.envs import TestEnv

# Custom Project Modules
from intersection_netw import UnsignalizedIntersectionNetwork, SCENARIO_CONFIGS
from intersection_env import MultiTaskIntersectionEnv
from multitask_dqn_model import MultiTaskDQN

# File and Directory Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "dqn_run_20260824_134534")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "scenario_g_left_frames")
VIDEO_NAME = os.path.join(SCRIPT_DIR, "scenario_g_left_demo.mp4")


def resolve_checkpoint(run_dir):
    """Finds the model weight file inside the target checkpoint folder."""
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Directory not found: {run_dir}")

    priority_files = [
        os.path.join(run_dir, "ckpt_final.pt"),
        os.path.join(run_dir, "best_model.pt"),
        os.path.join(run_dir, "model.pt"),
        os.path.join(run_dir, "model.pth"),
    ]
    for p in priority_files:
        if os.path.isfile(p):
            return p

    found = glob.glob(os.path.join(run_dir, "**/*.pt"), recursive=True) + \
            glob.glob(os.path.join(run_dir, "**/*.pth"), recursive=True)

    if not found:
        raise FileNotFoundError(f"No .pt or .pth weight files found inside: {run_dir}")

    found.sort(key=os.path.getmtime, reverse=True)
    return found[0]


def record_scenario_g_demo():
    TASK_CHOICE = "left"
    VIEW_ID = "View #0"
    FPS = 10

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load exact MultiTaskDQN Model & Weights
    ckpt_path = resolve_checkpoint(TARGET_CKPT_DIR)
    print(f"\n[+] Loading checkpoint from: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = MultiTaskDQN().to(device)

    checkpoint_data = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint_data, dict):
        if "model_state_dict" in checkpoint_data:
            policy.load_state_dict(checkpoint_data["model_state_dict"])
        elif "state_dict" in checkpoint_data:
            policy.load_state_dict(checkpoint_data["state_dict"])
        elif "model" in checkpoint_data:
            policy.load_state_dict(checkpoint_data["model"])
        else:
            policy.load_state_dict(checkpoint_data)
    else:
        policy.load_state_dict(checkpoint_data)
    policy.eval()

    # 2. Build Scenario g Geometry & Continuous Dynamic InFlows
    scenario_g_cfg = SCENARIO_CONFIGS["scenario_g"]

    run_seed = random.randint(1, 99999)
    sim_params = SumoParams(
        sim_step=0.1,
        render=True,
        restart_instance=True,
        seed=run_seed
    )

    # Aggressive control parameters for the RL ego agent:
    # speed_mode="aggressive" (or speed_mode=1) disables internal junction yielding and checks
    rl_car_following = SumoCarFollowingParams(
        accel=2.6,
        decel=4.5,
        min_gap=1.5,
        max_speed=18.0,
        speed_mode="aggressive"
    )

    # Background human driver behavior
    congested_car_following = SumoCarFollowingParams(
        accel=2.8,
        decel=4.5,
        sigma=0.5,
        speed_dev=0.2,
        min_gap=1.5,
        max_speed=18.0,
        speed_mode=1
    )

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="rl",
        num_vehicles=0,
        color="green",
        car_following_params=rl_car_following
    )
    vehicles.add(
        veh_id="human",
        num_vehicles=0,
        color="white",
        car_following_params=congested_car_following
    )

    inflows = InFlows()

    # Heavy oncoming stream from West (primary crossing conflict for left-turn)
    inflows.add(
        veh_type="human",
        edge="west_in",
        vehs_per_hour=1400,
        depart_lane="random",
        depart_speed=10.0
    )

    # Cross-traffic from North
    inflows.add(
        veh_type="human",
        edge="north_in",
        vehs_per_hour=1100,
        depart_lane="random",
        depart_speed=10.0
    )

    # Cross-traffic from South
    inflows.add(
        veh_type="human",
        edge="south_in",
        vehs_per_hour=900,
        depart_lane="random",
        depart_speed=10.0
    )

    # Controlled inflow on East
    inflows.add(
        veh_type="human",
        edge="east_in",
        vehs_per_hour=200,
        depart_lane="random",
        depart_speed=10.0
    )

    net_params = NetParams(
        inflows=inflows,
        additional_params=scenario_g_cfg
    )

    flow_network = UnsignalizedIntersectionNetwork(
        name="scenario_g_network",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=InitialConfig(),
        traffic_lights=TrafficLightParams()
    )

    base_flow_env = TestEnv(
        env_params=EnvParams(horizon=500),
        sim_params=sim_params,
        network=flow_network
    )

    env = MultiTaskIntersectionEnv(base_flow_env)

    # 3. Reset Environment & Initialize Visuals
    print(f"[+] Initializing Scenario g (Task: '{TASK_CHOICE}', Seed: {run_seed})...")
    obs, info = env.reset(options={"task": TASK_CHOICE})

    traci_api = env.flow_env.k.kernel_api
    ego_id = env._current_ego_id

    # Enforce aggressive speed mode directly on the ego vehicle instance via TraCI
    # 0 = completely unregulated, 1 = no yield/emergency braking (standard aggressive)
    if ego_id in traci_api.vehicle.getIDList():
        traci_api.vehicle.setSpeedMode(ego_id, 1)

    # Center camera on the ego car
    traci_api.gui.setZoom(VIEW_ID, 450)
    if ego_id in traci_api.vehicle.getIDList():
        traci_api.gui.trackVehicle(VIEW_ID, ego_id)

    step = 0
    done = False
    task_g_tensor = torch.tensor(env.active_g, dtype=torch.float32, device=device).unsqueeze(0)

    # Allow vehicles to populate the conflict lanes before frame capture starts
    WARMUP_STEPS = 45
    print(f"[+] Running {WARMUP_STEPS} warm-up steps to flood Scenario g intersection...")
    for _ in range(WARMUP_STEPS):
        obs, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break

    print("[+] Recording simulation frames...")
    try:
        while not done and step < 400:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Model Forward Pass & Action Selection via masked_q
            with torch.no_grad():
                r_tensor = policy(obs_tensor)
                q_values = MultiTaskDQN.masked_q(r_tensor, task_g_tensor)
                action = torch.argmax(q_values, dim=1).item()

            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            # Capture frame screenshot
            frame_path = os.path.join(OUTPUT_DIR, f"frame_{step:05d}.png")
            traci_api.gui.screenshot(VIEW_ID, frame_path)
            step += 1

        print(f"[+] Simulation ended at step {step}.")
        print(f"Outcome: Success={step_info.get('is_success')}, Collision={step_info.get('is_collision')}")

    finally:
        env.flow_env.terminate()

    # 4. Compile Video via FFmpeg
    print("[+] Stitching frames into MP4 video...")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-r", str(FPS),
        "-i", os.path.join(OUTPUT_DIR, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        VIDEO_NAME
    ]
    res = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0:
        print(f"\n[✓] Demo video saved successfully:\n    {VIDEO_NAME}\n")
    else:
        print("\nFFmpeg error:")
        print(res.stderr)


if __name__ == "__main__":
    record_scenario_g_demo()