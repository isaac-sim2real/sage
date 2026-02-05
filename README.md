<h1 align="center"> SAGE: Sim2Real Actuator Gap Estimator </h1>

<div align="center">

[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-b.svg)](https://isaac-sim.github.io/IsaacLab/v2.2.0/index.html)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-b.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/index.html)
[![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

</div>

## Overview

SAGE (Sim2Real Actuator Gap Estimator) is a comprehensive toolkit for analyzing the differences between simulated and real robot joint motions. This project provides systematic tools for measuring, visualizing, and understanding sim-to-real gaps in robotic systems, enabling researchers and engineers to quantify and improve the transfer of robot behaviors from simulation to reality.

SAGE combines:

- **Isaac Sim simulation** for physics-based robot motion execution
- **Multi-metric evaluation** with statistical analysis and visualization
- **Real robot integration** for comprehensive sim-to-real comparison

## Table of Contents

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Host Setup](#host-setup)
  - [Docker Setup](#docker-setup)
- [Usage](#usage)
  - [Simulation Execution](#simulation-execution)
  - [Data Analysis](#data-analysis)
  - [Real Robot Integration](#real-robot-integration)
- [OSMO Workflow](#osmo-workflow)
- [Data Format](#data-format)
  - [Motion Files](#motion-files)
  - [Simulation Output](#simulation-output)
  - [Real Robot Output](#real-robot-output)
  - [Processed Sim2Real Datasets](#processed-sim2real-datasets)
- [Adding New Humanoids](#adding-new-humanoids)
  - [Input Preparation](#input-preparation)
  - [Simulation Setup](#simulation-setup)
  - [Real Robot Integration](#real-robot-integration)
- [Configuration](#configuration)
  - [Files and Directories](#files-and-directories)
  - [Robot Configuration Parameters](#robot-configuration-parameters)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Installation

### Prerequisites

- Ubuntu 22.04 LTS
- NVIDIA GPU with compatible drivers
- Python 3.10
- Isaac Sim 5.0.0
- Isaac Lab 2.2.0

> **Note:** If you are using the provided Docker image, you do not need to install Python, Isaac Sim, and Isaac Lab. These dependencies are pre-installed in the Docker image.

Clone the repository:

```bash
git clone https://github.com/isaac-sim2real/sage.git
cd sage
```

For the rest of installation, you can install the necessary dependencies either directly on your host machine, or by using the provided Docker image.

### Host Setup

Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to set up Isaac Sim and Isaac Lab, and then install the dependencies:

```bash
# Install dependencies
pip install -r requirements.txt
```

Set the PYTHONPATH to allow scripts to find the package:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

### Docker Setup

Alternatively if you have [Docker](https://docs.docker.com/engine/install/ubuntu/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed, you can build the Docker image without installing the dependencies on your host machine.

```bash
docker build -t sage .
```

then start the container:

```bash
xhost +
docker run --name isaac-lab --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -e DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v $HOME/.Xauthority:/root/.Xauthority \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   -v $(pwd):/app:rw \
   sage
```

and run the rest of the commands in the container.

## Usage

### Simulation Execution

Execute robot motions in Isaac Sim simulation:

```bash
${ISAACSIM_PATH}/python.sh scripts/run_simulation.py \
    --robot-name h1_2 \
    --motion-source amass \
    --motion-files motion_files/h1_2/amass \
    --valid-joints-file configs/h1_2_valid_joints.txt \
    --output-folder output \
    --fix-root \
    --physics-freq 200 \
    --render-freq 200 \
    --control-freq 50 \
    --kp 100 \
    --kd 2 \
    --headless
```

This should take about 20 minutes to complete. For debugging purposes, you can run the script without the `--headless` flag to visualize the simulation.

### Data Analysis

Generate comprehensive analysis reports comparing simulation and real robot data:

```bash
python scripts/run_analysis.py \
    --robot-name h1_2 \
    --motion-source amass \
    --motion-names "*" \
    --output-folder output \
    --valid-joints-file configs/h1_2_valid_joints.txt
```

**Outputs:**

- **Metrics Excel files** with RMSE, MAPE, correlation, cosine similarity
- **Visualization plots** for individual joint comparisons (position, velocity, torque)
- **Statistical boxplots** comparing simulation vs real robot performance

### Real Robot Integration

**Different robot manufacturers provide different control APIs and dependencies.** For example, Unitree robots require ROS2 and Unitree SDK, Realman robots require their proprietary `Robotic_Arm_Custom package`, while LeRobot SO-101 requires the `lerobot` package. See the robot-specific documentation below for detailed installation instructions.

**Supported robots:**

- **Unitree G1** and **H1-2** humanoid robots
- **Realman WR75S** dual-arm robot
- **LeRobot SO-101** follower arm

Use the unified script to collect motion data on real robots (with custom robot-specific parameters):

```bash
python scripts/run_real.py \
    --robot-name {g1|h12|realman|so101} \
    --motion-files 'path/to/your/motion_sequence.txt' \
    --output-folder 'path/to/your/output_folder'
```

For detailed setup instructions, usage examples, and robot-specific configurations, refer to:

- [UNITREE_REAL](docs/UNITREE_REAL.md) - Unitree G1 and H1-2 guide
- [REALMAN_REAL](docs/REALMAN_REAL.md) - Realman WR75S guide
- [LEROBOT_REAL](docs/LEROBOT_REAL.md) - LeRobot SO-101 guide

## OSMO Workflow

We have created an OSMO workflow for one-click submission of simulation and analysis tasks. The results will be collected in the form of OSMO datasets. Please refer to the OSMO documentation to onboard OSMO, and set up the NGC registry credentials:

```bash
osmo credential set my-ngc-cred \
    --type REGISTRY \
    --payload registry=nvcr.io \
    username='$oauthtoken' \
    auth=<ngc_api_key>
```

The workflow can be submitted using the following command:

```bash
osmo workflow submit osmo_workflow.yaml
```

The results can be downloaded using the following command:

```bash
osmo dataset download sage ./
```

> **Note:** The `osmo_workflow.yaml` file is configured to run simulations for all 3 robots (h1_2, g1, wr75s) with all AMASS motion files on the A40 node of the OSMO platform. If necessary, you can modify this configuration file to suit different resource requirements or to run a subset of robots/motion files.

## Data Format

### Motion Files

Motion files contain joint trajectories retargeted to specific robots. Located in `motion_files/{robot_name}/{source}/`.

**Format:**

- **Line 1**: Joint names (comma-separated)
- **Line 2+**: Joint angles in radians (comma-separated)

**Motion Sources:**

- **AMASS**: Motion capture data from [AMASS Dataset](https://amass.is.tue.mpg.de/)
- **Retargeting**: Convert motion capture to robot morphology. Various retargeting methods exist, e.g., see [Human2Humanoid](https://github.com/LeCAR-Lab/human2humanoid?tab=readme-ov-file#motion-retargeting)

### Simulation Output

Generated in `output/sim/{robot_name}/{source}/{motion_name}/`:

- **control.csv**: Command positions sent to robot (**radians**)
- **state_motor.csv**: Actual joint states (positions, velocities, torques) (**radians**)
- **joint_list.txt**: Joint configuration

**CSV Format:**

```csv
type,timestamp,positions,velocities,torques
CONTROL/STATE_MOTOR,0.0,"[angle1, angle2, ...]","[vel1, vel2, ...]","[torque1, torque2, ...]"
```

### Real Robot Output

Generated in `output/real/{robot_name}/{source}/{motion_name}/`:

- **control.csv**: Commands sent to real robot (**radians**)
- **state_motor.csv**: Measured joint states (**radians**)
- **state_base.csv**: IMU/base measurements
- **event.csv**: Event timestamps

**Key Differences from Simulation:**

- Timestamps in microseconds (vs. seconds)
- Additional columns in state_motor.csv: temperatures, currents
- Type names: `Control/StateMotor` (vs. `CONTROL/STATE_MOTOR`)
- Irregular timing due to real-world constraints

### Processed Sim2Real Datasets

After collecting both simulation and real robot data pairs, we process them into structured datasets suitable for training sim2real gap compensation models. These datasets align temporal sequences and provide paired observations for machine learning approaches.

The complete dataset containing both Unitree and RealMan robot data is available for download: [PKU Disk Link](https://disk.pku.edu.cn/link/AA479C804E805D4B26B5E5C544A6EE57F8).

#### Unitree Dataset

This dataset captures complex upper-body motions of the H1-2 humanoid robot under varying payload conditions (0 kg, 1 kg, 2 kg, and 3 kg). The motions are adapted from the open-source AMASS dataset and carefully post-processed to ensure reliable execution on the real robot. Each trajectory includes corresponding simulation replays, providing paired sim-real data for gap analysis and compensation model training.

**Data Variants:**
- **Standard split**: Training and test sets with upper-body motions (`train.npz`, `test.npz`)
- **Gait variations**: Upper-body motions paired with different lower-body gaits (locomotion, squatting, upper-only) to enhance data diversity
- **Whole-body extension**: A subset featuring full-body coordinated motions for comprehensive sim2real research

#### RealMan Dataset

This dataset contains data collected from four Realman WR75S robotic arms (`robot1-4`), each tested under four payload conditions (0-3 kg). The multi-robot, multi-payload structure enables cross-robot generalization studies and payload adaptation analysis.

## Adding New Humanoids

This section describes how to extend the framework to support new humanoid robots.

### Input Preparation

**Motion Files:**
Prepare retargeted motion files for your specific humanoid robot:

- Follow the motion file format described in [Motion Files](#motion-files)
- Ensure joint names match your robot's kinematic chain
- Place files in `motion_files/{robot_name}/{source}/`
- Joint angles must be in radians

### Simulation Setup

This section gives a general idea to add simulation support for a new humanoid. For detailed instructions, please refer to the [walkthrough](docs/NEW_ROBOT.md):

**1. Prepare USD Assets**

- Add your robot's USD file to `assets/` directory
- Ensure proper joint naming and hierarchy

**2. Update Simulation Code**

- Modify `joint_motion_gap/simulation.py`:
  - Add robot name to supported robots list
  - Configure USD path and prim path for your robot
  - Adjust any robot-specific simulation parameters

**3. Add Joint Configuration**

- Create `configs/{robot_name}_joints.yaml` with complete joint list
- Create `configs/{robot_name}_valid_joints.txt` with motion-relevant joints
- Ensure joint names match both USD asset and motion files

### Real Robot Integration

This section provides methods for extending our current real data collection pipeline to new robots. To facilitate usage across more robot models, we built the current transmission pipeline using ROS.

**1. Identify Your IP and Port**

- Place the ROS transmission files that exactly match your robot under the path `sage/Your_robot`
- Ensure the IP and message types are completely correct, and test them by creating nodes

**2. Build New Config**

- Refer to `sage/real_unitree/unitree_configs.py` to organize the required structure for the new robot model
- Be sure to confirm that the p_gains & d_gains for each joint are compatible with the physical robot, as they significantly impact data collection quality
- Place the new config under the path `sage/Your_robot`

**3. Update Main File**

- After completing the above tasks, navigate to `scripts/run_real.py` and update the relevant paths and tasks for your robot in the corresponding functions

## Configuration

### Files and Directories

- **Robot Asset Configurations**: `sage/assets.py`
- **Robot Assets**: `assets/{robot_name}/`
- **Valid Joints**: `configs/{robot_name}_valid_joints.txt`
- **Motion Files**: `motion_files/{robot_name}/{source}/`

### Robot Configuration Parameters

Each robot in `assets.py` can specify:

- `usd_path`: Path to robot USD file
- `offset`: (x, y, z) spawn position
- `default_kp`: Default stiffness for PD controller
- `default_kd`: Default damping gain for PD controller
- `default_control_freq`: Default control frequency (Hz) for motion playback

> **Parameter Priority System:**
> Control parameters (`kp`, `kd`, `control_freq`, `original_control_freq`) are resolved in the following order:
>
> 1. **Command-line arguments** - Explicitly provided via `--kp`, `--kd`, `--control-freq`, `--original-control-freq`
> 2. **Robot configuration** - Defaults from `assets.py` if arguments are not provided
> 3. **Error** - Raises an error if neither source is configured
>
> This design allows robot-specific defaults while enabling per-run customization.

**Supported Robots:** h1_2, g1, wr75s

**Motion Sources:** amass

## Contributors

See [Contributors](docs/CONTRIBUTOR.md) for the list of project contributors.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on code style and pull request process.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{sage-2025,
  title={SAGE: Sim2Real Actuator Gap Estimator},
  author={SAGE Team},
  year={2025},
  url={https://github.com/isaac-sim2real/sage}
}
```
