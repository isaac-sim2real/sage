#!/usr/bin/env python3
"""
Simple test script for GapONet actuator.

This script verifies that the GapONet actuator can be loaded and used in IsaacLab.
It creates a minimal simulation with a robot using the GapONet actuator.

Usage:
    cd /home/xiangx/NVIDIA_XIANG/code/SAGEPlugin/sage
    ../IsaacLab/isaaclab.sh -p scripts/test_gaponet.py --headless
"""

import argparse
import sys

# Parse arguments before importing Isaac Sim
parser = argparse.ArgumentParser(description="Test GapONet actuator")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument(
    "--model_path",
    type=str,
    default="models/policy.pt",
    help="Path to GapONet model (policy.pt)",
)
args, unknown = parser.parse_known_args()

# Launch Isaac Sim (must be done before importing isaaclab modules)
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import os  # noqa: E402

import torch  # noqa: E402
from isaaclab.actuators import GapONetActuatorCfg  # noqa: E402


def main():
    """Test GapONet actuator loading and basic functionality."""

    print("=" * 80)
    print("GapONet Actuator Test")
    print("=" * 80)

    # Step 1: Check if model file exists
    print(f"\n[1/5] Checking model file: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"❌ ERROR: Model file not found at: {args.model_path}")
        print("\nPlease ensure you have:")
        print("  1. Trained a GapONet model")
        print("  2. Exported it using inference_jit.py --export")
        print("  3. Copied it to the correct location")
        print("\nExample:")
        print("  cd /home/xiangx/NVIDIA_XIANG/code/SAGEPlugin/sage")
        print("  mkdir -p models")
        print("  cp /path/to/gaponet/model/policy.pt models/policy.pt")
        simulation_app.close()
        sys.exit(1)

    print(f"✓ Model file found: {os.path.getsize(args.model_path) / 1024 / 1024:.2f} MB")

    # Step 2: Import GapONet actuator
    print("\n[2/5] Importing GapONet actuator...")
    try:
        import isaaclab.actuators  # noqa: F401

        print("✓ GapONetActuator imported successfully")
    except ImportError as e:
        print(f"❌ ERROR: Failed to import GapONetActuator: {e}")
        simulation_app.close()
        sys.exit(1)

    # Step 3: Create GapONet configuration
    print("\n[3/5] Creating GapONet configuration...")
    try:
        gaponet_cfg = GapONetActuatorCfg(
            joint_names_expr=[".*"],  # All joints
            network_file=args.model_path,
            action_scale=1.0,
            stiffness=0.0,
            damping=0.0,
            # DeepONet parameters
            sensor_dim=20,
            num_sensor_positions=20,
            model_history_length=4,
            model_history_dim=30,
            step_dt=0.005,
        )
        print("✓ Configuration created")
        print(f"  - Model file: {gaponet_cfg.network_file}")
        print(f"  - Action scale: {gaponet_cfg.action_scale}")
        print(f"  - History length: {gaponet_cfg.model_history_length}")
        print(f"  - History dim: {gaponet_cfg.model_history_dim}")
    except Exception as e:
        print(f"❌ ERROR: Failed to create configuration: {e}")
        simulation_app.close()
        sys.exit(1)

    # Step 4: Load the model
    print("\n[4/5] Loading GapONet model...")
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {device}")

        model = torch.jit.load(args.model_path, map_location=device)
        model.eval()
        print("✓ Model loaded successfully")

        # Test model inference with dummy data
        print("\n[5/5] Testing model inference...")
        batch_size = args.num_envs

        # Create dummy inputs
        model_obs = torch.randn(batch_size, 130, device=device)  # [10 pos + 120 history]
        branch_obs = torch.randn(batch_size, 400, device=device)  # [sensor data]
        trunk_obs = torch.randn(batch_size, 10, device=device)  # [target pos]

        with torch.no_grad():
            output = model(model_obs, branch_obs, trunk_obs)

        print("✓ Model inference successful")
        print(
            f"  - Input shapes: model_obs={model_obs.shape}, branch_obs={branch_obs.shape}, trunk_obs={trunk_obs.shape}"
        )
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    except Exception as e:
        print(f"❌ ERROR: Failed to load or test model: {e}")
        import traceback

        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)

    # Summary
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print("\nGapONet actuator is ready to use!")
    print("\n🚀 Next: Run GapONet Simulation")
    print("=" * 80)
    print()

    simulation_app.close()


if __name__ == "__main__":
    main()
