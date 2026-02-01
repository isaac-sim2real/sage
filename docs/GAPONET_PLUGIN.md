# GapONet Plugin for SAGE + IsaacLab

## Overview

GapONet Plugin integrates DeepONet neural network actuator into SAGE's IsaacLab simulation environment for sim-to-real transfer and motion tracking enhancement.

## Quick Start

**Prerequisites**: 
- All commands assume you are in the `sage/` directory
- Replace `<ISAACLAB_PATH>` with your actual IsaacLab installation path

### 1. Test GapONet Module (Optional)

```bash
<ISAACLAB_PATH>/isaaclab.sh -p scripts/test_gaponet.py \
    --model_path models/policy.pt \
    --headless
```

**Expected Output**: `✅ All tests passed!`

### 2. Run Baseline (No GapONet)

```bash
<ISAACLAB_PATH>/isaaclab.sh -p scripts/run_simulation_gaponet.py \
    --robot-name h1_2 \
    --motion-files motion_files/h1_2/amass/0-wave_both02_poses_action_sequence.txt \
    --valid-joints-file configs/h1_2_valid_joints.txt \
    --output-folder output/sim_baseline \
    --fix-root \
    --physics-freq 200 \
    --control-freq 50 \
    --original-control-freq 30 \
    --kp 100 \
    --kd 2 \
    --headless
```

### 3. Run GapONet Mode

```bash
<ISAACLAB_PATH>/isaaclab.sh -p scripts/run_simulation_gaponet.py \
    --robot-name h1_2 \
    --motion-files motion_files/h1_2/amass/0-wave_both02_poses_action_sequence.txt \
    --valid-joints-file configs/h1_2_valid_joints.txt \
    --output-folder output/sim_gaponet \
    --gaponet-model models/policy.pt \
    --use-gaponet \
    --fix-root \
    --physics-freq 200 \
    --control-freq 50 \
    --original-control-freq 30 \
    --kp 100 \
    --kd 2 \
    --headless
```

### 4. Compare Performance

After running both baseline and GapONet simulations, use the advanced comparison tool:

```bash
python scripts/compare_gaponet.py \
    --baseline output/sim_baseline/h1_2/0-wave_both02_poses_action_sequence \
    --gaponet output/sim_gaponet/h1_2/0-wave_both02_poses_action_sequence \
    --output output/comparison_results
```

#### Output Files

The comparison tool generates:

```
comparison_results/
├── metrics.json                    # Detailed metrics for all joints
├── summary.json                    # Aggregated statistics
├── rmse_comparison.png             # Per-joint RMSE visualization
└── timeseries_<joint>.png          # Example time series plot
```

#### Metrics Calculated

The tool computes **comprehensive metrics** for positions, velocities, and torques:

**1. Tracking Accuracy**
- **RMSE** (Root Mean Squared Error): Overall tracking error
- **MAE** (Mean Absolute Error): Average tracking error
- **Max Absolute Error**: Worst-case tracking error

**2. Waveform Similarity**
- **Correlation**: How similar are the waveforms? (closer to 1.0 = better)
- **Cosine Similarity**: Directional alignment (closer to 1.0 = better)
- **Max Lag**: Time delay between signals (closer to 0 = better)

**3. Control Effort** (for torques)
- **RMS Torque**: Root mean squared torque values
- **RMS Change %**: Percentage change in control effort

**4. Smoothness** (for velocities and torques)
- **Smoothness Mean/Std**: Measures control smoothness


## Parameters

### Required

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--robot-name` | Robot name | `h1_2` |
| `--motion-files` | Motion file or directory | `motion_files/h1_2/amass/` |
| `--output-folder` | Output directory | `output/sim_test` |

### GapONet Related

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--use-gaponet` | Enable GapONet | `False` |
| `--gaponet-model` | Model file path (.pt) | `None` (required if enabled) |
| `--gaponet-action-scale` | Delta position scale | `1.0` |

### Simulation

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--physics-freq` | Physics frequency (Hz) | `200` |
| `--control-freq` | Control frequency (Hz) | `50` |
| `--original-control-freq` | Original motion data frequency (Hz) | `None` (recommend 30) |
| `--kp` | PD stiffness | `100.0` |
| `--kd` | PD damping | `2.0` |
| `--fix-root` | Fix root joint | `False` |
| `--headless` | Headless mode | `False` |
| `--enable-realtime` | Real-time sync (for GUI) | `False` |



## How GapONet Works

### Network Architecture

**GapONet uses DeepONet architecture**:

```python
# Input (per control step, 50Hz = 20ms)
model_obs    = [10 positions + 120 history]  # [batch, 130]
branch_obs   = [sensor data (pos + vel)]     # [batch, 400]
trunk_obs    = [10 target positions]         # [batch, 10]

# Network Internal
sensor_pred = sensor_model(model_obs)         # Sim-to-Real domain adaptation
branch_obs[:, :N] = sensor_pred               # Replace first N sensor readings
normalized = normalizer(branch_obs, trunk_obs)
delta_actions = branch_net(branch_obs) * trunk_net(trunk_obs)  # DeepONet

# Output
delta_positions = delta_actions               # [batch, 10] position deltas
```

### Control Loop

```python
# Per physics step (200Hz = 5ms)
for step in physics_steps:
    if step % divisor == 0:  # Control frequency (50Hz)
        # 1. Get current state
        q_current = robot.data.joint_pos
        dq_current = robot.data.joint_vel
        q_target = motion_data[frame_idx]
        
        # 2. GapONet inference (if enabled)
        if use_gaponet:
            delta = gaponet_actuator(q_current, dq_current, q_target)
            q_corrected = q_target + action_scale * delta
        else:
            q_corrected = q_target
        
        # 3. Send position target
        robot.set_joint_position_target(q_corrected)
    
    # 4. PhysX automatically executes PD control (C++ layer, high stability)
    # tau = Kp * (q_corrected - q_current) + Kd * (0 - dq_current)
    
    # 5. Step simulation
    sim.step()
```


## Code Structure

### Core Files

```
sage/
├── scripts/
│   ├── run_simulation_gaponet.py   # Main script
│   └── test_gaponet.py             # Test script
│   └── compare_gaponet.py             # Compare script
├── sage/
│   └── simulation_lab.py           # GapONetBenchmark class
├── docs/
│   └── GAPONET_PLUGIN.md           # This document
└── models/
    └── policy.pt                   # GapONetBenchmark model

IsaacLab/source/isaaclab/isaaclab/actuators/
├── actuator_gaponet.py             # GapONetActuator implementation
├── actuator_cfg.py                 # GapONetActuatorCfg configuration
└── __init__.py                     # Export GapONetActuator
```
