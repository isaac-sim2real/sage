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
  - [Isaac Lab Setup](#isaac-lab-setup)
  - [Package Installation](#package-installation)
  - [Environment Setup](#environment-setup)
- [Usage](#usage)
  - [Simulation Execution](#simulation-execution)
  - [Data Analysis](#data-analysis)
  - [Real Robot Integration](#real-robot-integration)
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
- Python 3.10
- NVIDIA GPU with compatible drivers
- Isaac Sim 5.0.0
- Isaac Lab 2.2.0

### Isaac Lab Setup

Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to set up Isaac Sim and Isaac Lab.

### Package Installation

```bash
# Clone the repository
git clone https://github.com/isaac-sim2real/sage.git
cd sage

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Set the PYTHONPATH to allow scripts to find the package:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

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

**Different robot manufacturers provide different control APIs**, so real code need extra package requirements.

Both **Unitree G1 and H1-2** have been tested. Also, **Realman WR75S** has been tested. Detailed guidelines can be found in [UNITREE_REAL.md](docs/UNITREE_REAL.md) and [REALMAN_REAL.md](docs/REALMAN_REAL.md).


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

**Available Processed Datasets:**

| Robot Name    | Dataset Link                                                                |
| ------------- | --------------------------------------------------------------------------- |
| unitree_h1_2  | [Unitree H1-2 Sim2Real Dataset](https://example.com/unitree_h1_2_dataset)   |
| realman_wr75s | [Realman WR75S Sim2Real Dataset](https://example.com/realman_wr75s_dataset) |

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

**Status:** TBD (To Be Determined)

Real robot integration steps will be documented as the framework evolves to support additional robots.

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
