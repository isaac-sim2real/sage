# Realman Robot Motion Data Collection

This project provides tools for collecting and processing motion data from Realman humanoid dual-arm robots. It supports loading motion sequences from TXT files and collecting real-time robot state data during motion execution.

## Installation

1. **Create and activate a Conda environment:**
    ```bash
    conda create -n realman python=3.9
    conda activate realman
    ```
2. **Install required packages:**
    ```bash
    pip install Robotic_Arm_Custom
    pip install numpy matplotlib h5py joblib scipy pandas
    ```

## Usage

### Execute Motion and Collect Data

```bash
python real_realman.py
```

Or use the main function:

```python
from real_realman import real_realman_main

real_realman_main(
    txt_file_path='path/to/motion.txt',
    save_data_path='path/to/output'
)
```

## Input Format

### TXT Motion Files
Motion files should be in CSV format:
```
L_joint1,L_joint2,L_joint3,L_joint4,L_joint5,L_joint6,L_joint7,R_joint1,R_joint2,R_joint3,R_joint4,R_joint5,R_joint6,R_joint7
0.1,0.2,0.3,0.4,0.5,0.6,0.7,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7
...
```

- **First row**: Joint names (7 left arm + 7 right arm joints)
- **Subsequent rows**: Joint angles in radians for each motion frame
- **Frequency**: 50Hz

## Output

The system generates CSV files for each motion:

```
output/
└── motion_name/
    ├── joint_list.txt      # List of joint names
    ├── control.csv         # Command data sent to robot
    ├── event.csv          # Motion events and timestamps
    └── state_motor.csv    # Actual robot state data
```

## Robot Configuration

Default network settings:
- **Left arm**: IP 192.168.1.188, Port 8080
- **Right arm**: IP 192.168.1.188, Port 8576
- **Control Frequency**: 200Hz
- **Motion Frequency**: 50Hz
