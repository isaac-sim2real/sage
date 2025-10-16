# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse

from joint_motion_gap.simulation import JointMotionBenchmark, get_motion_files, get_motion_name, log_message


def main():
    parser = argparse.ArgumentParser(description="Benchmark joint motion in Isaac Sim")
    parser.add_argument("--robot-name", type=str, required=True, help="Robot name (h1_2)")
    parser.add_argument(
        "--motion-source",
        type=str,
        choices=["amass"],
        required=True,
        help="Source of the motion",
    )
    parser.add_argument(
        "--motion-files",
        type=str,
        required=True,
        help="Path to motion file(s) or directory. Can be a single file, multiple files "
        "(comma-separated), or a directory.",
    )
    parser.add_argument("--valid-joints-file", type=str, required=False, default=None, help="Path to valid joints file")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to output folder")
    parser.add_argument("--fix-root", action="store_true", default=False, help="Fix root joint")
    parser.add_argument("--physics-freq", type=int, default=200, help="Physics timestep frequency")
    parser.add_argument("--render-freq", type=int, default=200, help="Render timestep frequency")
    parser.add_argument("--control-freq", type=int, default=50, help="Control timestep frequency")
    parser.add_argument(
        "--original-control-freq", type=int, default=None, help="Original control frequency of motion files"
    )
    parser.add_argument("--kp", type=int, default=None, help="Override default joint stiffness if provided")
    parser.add_argument("--kd", type=int, default=None, help="Override default joint damping if provided")
    parser.add_argument("--solver-type", type=int, default=1, help="Solver type TGS = 1 PGS = 0")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--record-video", action="store_true", help="Record video")

    # Parse command line arguments
    args = parser.parse_args()

    # Initialize the SimulationApp before importing any omni modules
    from isaacsim.simulation_app import SimulationApp

    app = SimulationApp({"headless": args.headless})

    # Now import Isaac Sim modules after SimulationApp initialization
    import carb  # noqa: E402
    import isaaclab.sim.schemas as schemas  # noqa: E402
    from isaacsim.core.api import World  # noqa: E402
    from isaacsim.core.prims import Articulation  # noqa: E402
    from isaacsim.core.utils.prims import create_prim  # noqa: E402
    from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage  # noqa: E402
    from omni.kit.viewport.utility import capture_viewport_to_buffer  # noqa: E402

    # Set the Isaac Sim modules in the simulation module
    import joint_motion_gap.simulation as sim_module

    sim_module.schemas = schemas
    sim_module.World = World
    sim_module.Articulation = Articulation
    sim_module.add_reference_to_stage = add_reference_to_stage
    sim_module.create_new_stage = create_new_stage
    sim_module.capture_viewport_to_buffer = capture_viewport_to_buffer
    sim_module.create_prim = create_prim

    # Set carb module in the assets module
    import joint_motion_gap.assets as assets_module

    assets_module.carb = carb

    # Get list of motion files to process
    motion_files = get_motion_files(args.motion_files)
    if not motion_files:
        raise ValueError(f"No motion files found in {args.motion_files}")

    log_message(f"Found {len(motion_files)} motion files to process")

    # Initialize benchmark environment
    benchmark = JointMotionBenchmark(args)

    # Process each motion file
    for motion_file in motion_files:
        try:
            motion_name = get_motion_name(motion_file)
            log_message(f"################### PROCESSING {motion_file} ###################")
            # Set up and run benchmark for this motion
            benchmark.set_motion(motion_file, motion_name)
            benchmark.run_benchmark()

        except Exception as e:
            log_message(f"Error processing {motion_file}: {str(e)}")
            continue

    # Print joint properties summary at the end
    benchmark.log_joint_properties()

    app.close()


if __name__ == "__main__":
    main()
