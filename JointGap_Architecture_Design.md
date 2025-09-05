# JointGap Repository Architecture Design Document

## 1. Project Overview

JointGap is a repository that integrates simulation execution, real execution, and error measurement of robot joint motions. The design principle is to minimize changes to existing benchmark-joint-motion code while focusing on the h1_2 robot.

## 2. Simplified Architecture

### 2.1 Directory Structure (Minimal Adjustments Based on Existing Code)

```
JointGap/
├── jointgap/                      # Python package
│   ├── __init__.py
│   ├── simulation.py              # Simulation module (renamed from benchmark_joint_motion.py)
│   ├── analysis.py                # Analysis module (renamed from benchmark_metrics.py)
│   └── real.py                    # Real execution module (placeholder)
├── configs/                       # Configuration files (copied from benchmark-joint-motion)
│   ├── h1_2_joints.yaml
│   └── h1_2_valid_joints.txt
├── motion_files/                  # Motion files (copied from benchmark-joint-motion)
│   └── h1_2/
│       ├── amass/
│       ├── captured/
│       └── crafted/
├── assets/                        # Asset files (copied from benchmark-joint-motion)
│   └── h1_2/
├── scripts/                       # Command line scripts
│   ├── run_simulation.py          # Wrapper for simulation.py
│   ├── run_analysis.py            # Wrapper for analysis.py
│   └── run_real.py                # placeholder
├── output/                        # Output directory (keep existing structure)
│   ├── sim/
│   ├── real/
│   └── metrics/
├── .pre-commit-config.yaml        # Copied from COMPASS
├── pyproject.toml                 # Copied from benchmark-joint-motion and adjusted
├── requirements.txt               # Copied from benchmark-joint-motion
├── README.md                      # Rewritten following COMPASS format
├── LICENSE                        # Copied from COMPASS
├── CHANGELOG.md                   # Following COMPASS format
└── CONTRIBUTING.md                # Copied from COMPASS
```

### 2.2 Code Migration Strategy (Minimal Changes)

#### 2.2.1 Simulation Module (`jointgap/simulation.py`)
```python
# Directly copy benchmark_joint_motion.py with only these changes:
# 1. Modify import paths to adapt to new directory structure
# 2. Keep all existing functionality and class names unchanged
# 3. Only modify default paths to point to new directory structure

class JointMotionBenchmark:
    # Keep all original code, only modify path-related parts
    def __init__(self, args):
        # Original code...
        # Only modify here: config file path
        config_file = os.path.join("configs", f"{self.robot_name}_valid_joints.txt")
        # Asset file path
        self.usd_path = os.path.join("assets", "robots.usd")
```

#### 2.2.2 Analysis Module (`jointgap/analysis.py`)
```python
# Directly copy benchmark_metrics.py with only these changes:
# 1. Modify import paths
# 2. Keep all existing functionality and class names unchanged
# 3. Modify default configuration paths

class RobotDataComparator:
    # Keep all original code, only modify path-related parts
    def _load_joint_config(self):
        config_path = Path("configs") / f"{self.robot_name}_joints.yaml"
        # Keep the rest unchanged...
```

#### 2.2.3 Real Execution Module (`jointgap/real.py`)
```python
# Simple placeholder implementing the same interface
class RealMotionExecutor:
    def __init__(self, args):
        # Keep the same parameter structure as JointMotionBenchmark
        pass
    
    def execute_motion(self, motion_file, motion_name):
        # placeholder - output same format data to output/real/ directory
        raise NotImplementedError("Real execution not implemented yet")
```

### 2.3 Script Wrapper Layer

#### 2.3.1 `scripts/run_simulation.py`
```python
# Simple wrapper for original functionality
import sys
from jointgap.simulation import main

if __name__ == "__main__":
    main()  # Directly call the original main function
```

#### 2.3.2 `scripts/run_analysis.py`
```python
# Simple wrapper for original functionality
import sys
from jointgap.analysis import main

if __name__ == "__main__":
    main()  # Directly call the original main function
```

#### 2.3.3 `scripts/run_real.py`
```python
# Simple placeholder for future real execution
import sys
from jointgap.real import RealMotionExecutor

def main():
    print("Real execution not implemented yet")
    print("This script is a placeholder for future real robot execution")

if __name__ == "__main__":
    main()
```

### 2.4 Configuration and Data Migration

#### 2.4.1 Direct File Copy
```bash
# Copy directly from benchmark-joint-motion without modification
cp benchmark-joint-motion/configs/h1_2_joints.yaml configs/
cp benchmark-joint-motion/configs/h1_2_valid_joints.txt configs/
cp -r benchmark-joint-motion/motion_files/h1_2/ motion_files/
cp -r benchmark-joint-motion/assets/h1_2/ assets/
```

#### 2.4.2 Maintain Existing Data Formats
- Motion files: Keep existing format completely
- Output CSV: Keep existing format completely  
- Configuration files: Keep existing format completely

### 2.5 Documentation and Standards (Following COMPASS)

#### 2.5.1 Direct Copy from COMPASS
```bash
cp COMPASS/LICENSE ./
cp COMPASS/.pre-commit-config.yaml ./
cp COMPASS/CONTRIBUTING.md ./
```

#### 2.5.2 Adjust Project-Specific Files
- `README.md`: Follow COMPASS format but content for JointGap
- `pyproject.toml`: Based on benchmark-joint-motion, add COMPASS style configurations
- `CHANGELOG.md`: Follow COMPASS format

## 3. Implementation Steps

### 3.1 Step 1: Create Basic Structure
1. Create directory structure
2. Copy COMPASS standard files (LICENSE, CONTRIBUTING, etc.)
3. Copy benchmark-joint-motion configuration and data files

### 3.2 Step 2: Code Migration (Minimal Changes)
1. Copy `benchmark_joint_motion.py` → `jointgap/simulation.py`
2. Copy `benchmark_metrics.py` → `jointgap/analysis.py`  
3. Only modify file path related parts
4. Create `jointgap/real.py` placeholder

### 3.3 Step 3: Script Wrapping
1. Create simple wrapper scripts (run_simulation.py, run_analysis.py)
2. Create placeholder script (run_real.py)
3. Ensure command line interface remains compatible

### 3.4 Step 4: Documentation and Testing
1. Write README (following COMPASS format)
2. Test that existing functionality works properly
3. Set up pre-commit hooks
