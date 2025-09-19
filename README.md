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
git clone <repository-url>
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
