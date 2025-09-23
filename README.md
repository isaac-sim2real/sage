<h1 align="center"> Joint Motion Gap Analysis Framework </h1>

<div align="center">

[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-b.svg)](https://isaac-sim.github.io/IsaacLab/v2.1.0/index.html)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-b.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
[![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

</div>

## Overview

Joint Motion Gap Analysis Framework is a comprehensive toolkit for analyzing the differences between simulated and real robot joint motions. This project provides systematic tools for measuring, visualizing, and understanding sim-to-real gaps in robotic systems, enabling researchers and engineers to quantify and improve the transfer of robot behaviors from simulation to reality.

Joint Motion Gap Framework combines:
- **Isaac Sim simulation** for physics-based robot motion execution
- **Multi-metric evaluation** with statistical analysis and visualization
- **Real robot integration** for comprehensive sim-to-real comparison

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Isaac Lab Setup](#isaac-lab-setup)
  - [Package Installation](#package-installation)
- [Usage](#usage)
  - [Simulation Execution](#simulation-execution)
  - [Data Analysis](#data-analysis)
  - [Real Robot Integration (Coming Soon)](#real-robot-integration-coming-soon)
- [Data Format](#data-format)
  - [Motion Files](#motion-files)
  - [Simulation Output](#simulation-output)
  - [Real Robot Output](#real-robot-output)
  - [Processed Sim2Real Datasets](#processed-sim2real-datasets)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Installation

### Prerequisites

- Ubuntu 22.04 LTS
- Python 3.10
- NVIDIA GPU with compatible drivers
- Isaac Sim 4.5.0

### Isaac Lab Setup

Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to set up Isaac Sim and Isaac Lab.

### Package Installation

```bash
# Clone the repository
git clone https://github.com/isaac-sim2real/joint-motion-gap.git
cd joint-motion-gap

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

### Real Robot Integration (Coming Soon)

```bash
# Placeholder for future real robot execution
python scripts/run_real.py \
    --robot-name h1_2 \
    --motion-files motion_files/h1_2/amass \
    --output-folder output
```

## Data Format

### Motion Files

Motion files contain joint trajectories retargeted to specific robots. Located in `motion_files/{robot_name}/{source}/`.

**Format:**
- **Line 1**: Joint names (comma-separated)
- **Line 2+**: Joint angles in radians (comma-separated)

**Motion Sources:**
- **AMASS**: Motion capture data from [AMASS Dataset](https://amass.is.tue.mpg.de/)
- **Retargeting**: Convert motion capture to robot morphology using [Human2Humanoid](https://github.com/LeCAR-Lab/human2humanoid?tab=readme-ov-file#motion-retargeting)

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

| Robot Name | Dataset Link |
|------------|-------------|
| unitree_h1_2 | [Unitree H1-2 Sim2Real Dataset](https://example.com/unitree_h1_2_dataset) |
| realman_wr75s | [Realman WR75S Sim2Real Dataset](https://example.com/realman_wr75s_dataset) |

## Configuration

- **Robot Configurations**: `configs/{robot_name}_joints.yaml`
- **Valid Joints**: `configs/{robot_name}_valid_joints.txt`
- **Motion Files**: `motion_files/{robot_name}/{source}/`
- **Robot Assets**: `assets/{robot_name}/`

**Supported Robots:** h1_2

**Motion Sources:** amass

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on code style and pull request process.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{joint-motion-gap-2025,
  title={Joint Motion Gap Analysis Framework},
  author={JointMotionGap Team},
  year={2025},
  url={https://github.com/your-org/joint-motion-gap}
}
```
