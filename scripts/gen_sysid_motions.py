import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Configuration ---
FREQ = 100  # 100 Hz control loop
DT = 1.0 / FREQ
JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
HOME = [0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
PATH = "../motion_files/so101/custom/" 

# Position and Velocity Safety Limits
SAFE_LIMITS = {
    "Rotation": (-2.04, 2.04), "Pitch": (-1.82, 1.82), "Elbow": (-1.70, 1.70),
    "Wrist_Pitch": (-1.73, 1.73), "Wrist_Roll": (-2.94, 2.94), "Jaw": (0.00, 1.00)
}
MAX_VELOCITIES = {
    "Rotation": 1.5, "Pitch": 1.5, "Elbow": 2.5,
    "Wrist_Pitch": 3.0, "Wrist_Roll": 3.0, "Jaw": 2.0
}

def validate_trajectory(data):
    """Checks position limits and calculates velocity for hardware safety."""
    for row_idx in range(len(data)):
        current_pose = data[row_idx]
        for i, joint_name in enumerate(JOINTS):
            low, high = SAFE_LIMITS[joint_name]
            if not (low - 1e-5 <= current_pose[i] <= high + 1e-5):
                raise ValueError(f"POS ERROR: {joint_name} at row {row_idx} is {current_pose[i]:.4f}")
        if row_idx > 0:
            for i, joint_name in enumerate(JOINTS):
                velocity = abs(current_pose[i] - data[row_idx-1][i]) / DT
                if velocity > MAX_VELOCITIES[joint_name] + 0.05:
                    raise ValueError(f"VELOCITY ERROR: {joint_name} at row {row_idx} at {velocity:.2f} rad/s")

def generate_smooth_transition(start, end, duration_s):
    num_steps = int(duration_s * FREQ)
    t = np.linspace(0, np.pi, num_steps)
    interp = 0.5 * (1 - np.cos(t))
    return [((1 - a) * np.array(start) + a * np.array(end)).tolist() for a in interp]

def save_txt(filename, data):
    validate_trajectory(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pd.DataFrame(data, columns=JOINTS).to_csv(filename, index=False, float_format="%.6f")
    print(f"Validated and Saved: {filename}")

# --- 1. Friction & Gravity Map ---
def make_friction_gravity():
    traj = []
    for i in range(len(JOINTS)):
        t_pos = list(HOME); t_neg = list(HOME)
        t_pos[i] += 0.3; t_neg[i] -= 0.3
        traj += generate_smooth_transition(HOME, t_pos, 1.5)
        traj += generate_smooth_transition(t_pos, t_neg, 3.0)
        traj += generate_smooth_transition(t_neg, HOME, 1.5)
    save_txt(os.path.join(PATH, "friction_gravity.txt"), traj)

# --- 2. Inertia & Coupling ---
def make_inertia_coupling():
    traj = []; tucked = [0.0, 0.5, 0.0, 0.0, 0.0, 0.5]; ext = [0.0, 0.0, 1.0, 0.0, 0.0, 0.5]
    traj += generate_smooth_transition(HOME, tucked, 2.0)
    for _ in range(2):
        traj += generate_smooth_transition(tucked, [0.4, 0.5, 0.0, 0.0, 0.0, 0.5], 1.0)
        traj += generate_smooth_transition([0.4, 0.5, 0.0, 0.0, 0.0, 0.5], [-0.4, 0.5, 0.0, 0.0, 0.0, 0.5], 2.0)
        traj += generate_smooth_transition([-0.4, 0.5, 0.0, 0.0, 0.0, 0.5], tucked, 1.0)
    traj += generate_smooth_transition(tucked, ext, 2.0)
    for _ in range(2):
        traj += generate_smooth_transition(ext, [0.4, 0.0, 1.0, 0.0, 0.0, 0.5], 1.0)
        traj += generate_smooth_transition([0.4, 0.0, 1.0, 0.0, 0.0, 0.5], [-0.4, 0.0, 1.0, 0.0, 0.0, 0.5], 2.0)
        traj += generate_smooth_transition([-0.4, 0.0, 1.0, 0.0, 0.0, 0.5], ext, 1.0)
    traj += generate_smooth_transition(ext, HOME, 2.0)
    save_txt(os.path.join(PATH, "inertia_coupling.txt"), traj)

# --- 3. Actuator Bandwidth (Velocity-Aware) ---
def make_actuator_bandwidth():
    duration = 20.0
    num_steps = int(duration * FREQ)
    t = np.linspace(0, duration, num_steps)
    freq_ramp = np.linspace(0.1, 2.0, num_steps)
    phase = 2 * np.pi * np.cumsum(freq_ramp) * DT

    # scale amplitude to stay under 75% of max velocity
    dynamic_amp = np.minimum(0.15, (MAX_VELOCITIES["Elbow"] * 0.75) / (2 * np.pi * freq_ramp))
    oscillations = (np.clip(t / 4.0, 0, 1) * dynamic_amp) * np.sin(phase)
    traj = [HOME] * int(1.5 * FREQ)
    for val in oscillations:
        p = list(HOME); p[2] += val
        traj.append(p)
    traj += [HOME] * int(1.5 * FREQ)
    save_txt(os.path.join(PATH, "actuator_bandwidth.txt"), traj)

# --- 4. End Effector Dynamics ---
def make_end_effector():
    traj = []; w1 = [0.0, 0.0, 0.5, 0.3, 0.5, 0.5]; w2 = [0.0, 0.0, 0.5, -0.3, -0.5, 0.5]
    for _ in range(3):
        traj += generate_smooth_transition(HOME, w1, 1.0)
        traj += generate_smooth_transition(w1, w2, 2.0)
        traj += generate_smooth_transition(w2, HOME, 1.0)
    open_j = [0.0, 0.0, 0.5, 0.0, 0.0, 0.9]; closed_j = [0.0, 0.0, 0.5, 0.0, 0.0, 0.1]
    for _ in range(3):
        traj += generate_smooth_transition(HOME, open_j, 1.0)
        traj += generate_smooth_transition(open_j, closed_j, 1.0)
        traj += generate_smooth_transition(closed_j, HOME, 1.0)
    save_txt(os.path.join(PATH, "end_effector.txt"), traj)


if __name__ == "__main__":
    make_friction_gravity()
    make_inertia_coupling()
    make_actuator_bandwidth()
    make_end_effector()

    path = "../motion_files/so101/custom/"
    files = ["friction_gravity.txt", "inertia_coupling.txt", "actuator_bandwidth.txt", "end_effector.txt"]
    titles = ["1. Friction & Gravity", "2. Inertia & Coupling", "3. Actuator Bandwidth", "4. End Effector"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (file, title) in enumerate(zip(files, titles)):
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path): continue
        
        df = pd.read_csv(full_path)
        time = np.arange(len(df)) / FREQ
        for joint in JOINTS:
            axes[i].plot(time, df[joint], label=joint)
            
        axes[i].set_title(title)
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Position (Rad)")
        axes[i].grid(True, alpha=0.3)
        if i == 0: axes[i].legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.savefig("actual_sysid_file_plots.png")
