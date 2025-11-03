# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ast
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import signal
from scipy.spatial import distance

HAND_JOINT_NAMES = ["left_hand_joint", "right_hand_joint"]  # Will filter


class RobotDataProcessor:
    """Main class for processing robot data"""

    def __init__(self, robot_name, motion_source, motion_name, is_simulation=True, file_root=None):
        self.robot_name = robot_name
        self.motion_source = motion_source
        self.motion_name = motion_name
        self.file_path = f"{file_root}/{robot_name}/{motion_source}/{motion_name}"
        self.is_simulation = is_simulation

        # Set data format
        if is_simulation:
            self.use_radians = True
            self.use_seconds = True
        else:
            self.use_radians = True
            self.use_seconds = False

        # Load all the joints from the humanoid config file
        self.joint_config = self._load_joint_config()
        # Load the joints of interest for the motion
        self.joint_list = self._load_joint_list()
        self.data = self._load_robot_data(["control", "state_motor"])

    def _load_joint_config(self):
        """Load joint list from yaml configuration file"""
        config_path = Path(__file__).parent.parent / "configs" / f"{self.robot_name}_joints.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return tuple(config["joints"])

    def _load_joint_list(self):
        """Load joint list from file or default configuration and apply mask if provided"""
        joint_list_path = Path(f"{self.file_path}/joint_list.txt")

        if joint_list_path.is_file():
            with open(joint_list_path) as file:
                return tuple(line.strip().split("/")[-1] for line in file.readlines())
        else:
            return tuple(joint.strip().split("/")[-1] for joint in self.joint_config)

    def _load_robot_data(self, data_types):
        """Load and process robot data from files"""
        robot_data = {}
        initial_time = float("inf")

        # Read data for each type
        for data_type in data_types:
            data = pd.read_csv(f"{self.file_path}/{data_type}.csv")

            # Convert time units if needed
            if not self.use_seconds:
                data["timestamp"] = data["timestamp"] / 1e6

            initial_time = min(initial_time, data["timestamp"][0])
            robot_data[data_type] = data

        # Calculate relative timestamps
        for data in robot_data.values():
            data["time_since_zero"] = data["timestamp"] - initial_time
            data["time_since_last"] = data["timestamp"].diff()

        # Process DOF data
        dof_command = self._process_dof_data(robot_data["control"], self.joint_list, ["positions"], self.use_radians)
        dof_state = self._process_dof_data(
            robot_data["state_motor"], self.joint_list, ["positions", "velocities", "torques"], self.use_radians
        )

        return {"raw_data": robot_data, "dof_command": dof_command, "dof_state": dof_state}

    @property
    def real_event(self):
        event = pd.read_csv(f"{self.file_path}/event.csv")
        return dict(zip(event["event"], event["timestamp"] / 1e6))

    @property
    def raw_data(self):
        return self.data["raw_data"]

    @property
    def dof_command(self):
        return self.data["dof_command"]

    @property
    def dof_state(self):
        return self.data["dof_state"]

    def _preprocess_string(self, value):
        pattern = r"\b([a-zA-Z_][a-zA-Z_0-9]*)\b"
        return re.sub(pattern, r"'\1'", value)

    def _safe_str_to_list(self, value):
        try:
            preprocessed_value = self._preprocess_string(value)
            return ast.literal_eval(preprocessed_value)
        except (ValueError, SyntaxError):
            return "Invalid list string"

    def _safe_deg2rad(self, x, i):
        if pd.api.types.is_number(x[i]):
            return np.deg2rad(x[i])
        return x[i]

    def _process_dof_data(self, df, joint_list, keys=["positions"], is_rad=False):
        df_copy = df.copy()

        for k in keys:
            df_copy[k] = df_copy[k].apply(self._safe_str_to_list)

            key_df = pd.DataFrame(index=df.index)
            for i, j in enumerate(joint_list):
                if not is_rad and k in ["positions", "velocities"]:
                    key_df[f"{k}_{j}"] = df_copy[k].apply(lambda x: self._safe_deg2rad(x, i))
                else:
                    key_df[f"{k}_{j}"] = df_copy[k].apply(lambda x: x[i])

            df_copy = pd.concat([df_copy.drop(k, axis=1), key_df], axis=1)

        return df_copy

    @property
    def joints(self):
        return self.joint_list


class RobotDataComparator:
    """
    A class for comparing simulation and real robot data, computing metrics,
    and visualizing the differences between simulated and real robot behavior.
    """

    def __init__(self, robot_name, motion_source, motion_names, valid_joints_file, result_folder: str, sample_dt: int):
        """
        Initialize the RobotDataComparator with parameters for data comparison.

        Args:
            robot_name (str): Name of the robot (h1_2)
            motion_source (str): Source of the motion data (e.g., 'amass')
            motion_names (str): Either "*" to use all available motions, or comma-separated motion names
            valid_joints_file (str): Path to file containing joints to include in analysis.
                                   If None, will look for {robot_name}_valid_joints.txt in configs folder.
                                   If no valid joints file is found, all joints will be used.
            result_folder (str): Path to the folder containing simulation and real data
                                (expects 'sim' and 'real' subfolders)
            sample_dt (int): Time step in milliseconds for resampling data during comparison.
                            Data will be linearly interpolated at this time step.
        """

        self._robot_name = robot_name
        self._motion_source = motion_source
        self._result_folder = result_folder
        self._valid_joints_list = self._load_valid_joints(valid_joints_file, self._robot_name)
        self._valid_joints_list = [path.split("/")[-1] for path in self._valid_joints_list]

        if len(self._valid_joints_list) == 0:
            self._use_all_joints = True
        else:
            self._use_all_joints = False

        self._motion_names = self._process_motion_names(
            motion_names, self._result_folder, self._robot_name, self._motion_source
        )
        if len(self._motion_names) == 0:
            raise ValueError("Can't get any motion_name")

        self._sample_dt = sample_dt

        self._motion_data = {}
        for i, motion_name in enumerate(self._motion_names):
            # Create sim and real data processors
            sim_data = RobotDataProcessor(
                self._robot_name,
                self._motion_source,
                motion_name,
                is_simulation=True,
                file_root=f"{self._result_folder}/sim/",
            )

            real_data = RobotDataProcessor(
                self._robot_name,
                self._motion_source,
                motion_name,
                is_simulation=False,
                file_root=f"{self._result_folder}/real/",
            )

            self._motion_data[motion_name] = {"sim": sim_data, "real": real_data}

    def _get_joints_for_motion(self, motion_name):
        sim_data = self._motion_data[motion_name]["sim"]

        if self._use_all_joints:
            selected_joints = sim_data.joints
            print(f"  [INFO] Using all {len(selected_joints)} joints for motion {motion_name}")
        else:
            sim_joints = sim_data.joints
            selected_joints = list(set(self._valid_joints_list).intersection(set(sim_joints)))
            if not selected_joints:
                selected_joints = sim_joints
                print(
                    f"  [WARNING] Using all {len(selected_joints)} joints for motion {motion_name} "
                    f"(mask had no common joints)"
                )
            else:
                print(
                    f"  [SUCCESS] Using {len(selected_joints)} joints from mask intersection for motion {motion_name}"
                )

        return [j for j in selected_joints if j not in HAND_JOINT_NAMES]

    def _load_valid_joints(self, valid_joints_file=None, robot_name=None):
        """
        Load valid joints file

        Args:
            valid_joints_file (str): Path to the valid joints file
            robot_name (str): Robot name

        Returns:
            list: List of joint masks
        """
        if valid_joints_file and os.path.exists(valid_joints_file):
            with open(valid_joints_file, "r") as file:
                return [line.strip() for line in file.readlines()]

        default_valid_joints_file = os.path.join(os.getcwd(), f"configs/{robot_name}_valid_joints.txt")
        if os.path.exists(default_valid_joints_file):
            with open(default_valid_joints_file, "r") as file:
                return [line.strip() for line in file.readlines()]

        print(f"[WARNING] No joints mask file found for '{robot_name}'. Using all joints.")
        return []

    def _process_motion_names(self, motion_names_arg, result_folder, robot_name, motion_source):
        """
        Process the motion names argument and return a list of motion names to process.
        Ensures that motion names exist in both sim and real directories.

        Args:
            motion_names_arg (str): Motion names argument, either "*" or comma-separated names
            result_folder (str): Path to the output folder
            robot_name (str): Name of the robot
            motion_source (str): Source of the motion (e.g., "amass")

        Returns:
            list: List of motion names to process

        Raises:
            ValueError: If motion names don't match between sim and real directories,
                       or if specified motion names don't exist in both directories
        """
        # Construct paths to sim and real directories
        sim_path = os.path.join(result_folder, "sim", robot_name, motion_source)
        real_path = os.path.join(result_folder, "real", robot_name, motion_source)

        # Check if both directories exist
        if not os.path.exists(sim_path):
            raise ValueError(f"Simulation results directory does not exist: {sim_path}")
        if not os.path.exists(real_path):
            raise ValueError(f"Real results directory does not exist: {real_path}")

        # Get motion directories from both sim and real folders
        sim_motions = [d for d in os.listdir(sim_path) if os.path.isdir(os.path.join(sim_path, d))]
        real_motions = [d for d in os.listdir(real_path) if os.path.isdir(os.path.join(real_path, d))]

        if motion_names_arg == "*":
            # Check if motion names match between sim and real directories
            if set(sim_motions) != set(real_motions):
                sim_only = set(sim_motions) - set(real_motions)
                real_only = set(real_motions) - set(sim_motions)
                error_msg = "Motion names don't match between sim and real directories:\n"
                if sim_only:
                    error_msg += f"- In sim only: {', '.join(sim_only)}\n"
                if real_only:
                    error_msg += f"- In real only: {', '.join(real_only)}"
                raise ValueError(error_msg)

            # Use the intersection of motions from both directories
            motion_names = sorted(list(set(sim_motions).intersection(set(real_motions))))
            if not motion_names:
                raise ValueError(f"No motion directories found in both {sim_path} and {real_path}")
            print(
                f"[INFO] Found {len(motion_names)} motions: "
                f"{', '.join(motion_names[:3])}{'...' if len(motion_names) > 3 else ''}"
            )
        else:
            # Parse comma-separated motion names
            requested_motions = [name.strip() for name in motion_names_arg.split(",")]

            # Check if all requested motions exist in both sim and real directories
            missing_in_sim = [m for m in requested_motions if m not in sim_motions]
            missing_in_real = [m for m in requested_motions if m not in real_motions]

            if missing_in_sim or missing_in_real:
                error_msg = "Some requested motions are missing:\n"
                if missing_in_sim:
                    error_msg += f"- Missing in sim: {', '.join(missing_in_sim)}\n"
                if missing_in_real:
                    error_msg += f"- Missing in real: {', '.join(missing_in_real)}"
                raise ValueError(error_msg)

            motion_names = requested_motions
            print(
                f"[INFO] Using specified motions: {', '.join(motion_names[:3])}{'...' if len(motion_names) > 3 else ''}"
            )

        return motion_names

    def adjust_real_data_timing(self, dataframe, start_delay, end_time):
        """Adjust timestamps of real robot data

        Args:
            dataframe: Original data with timestamps
            start_delay: Time delay to subtract from timestamps
            end_time: Maximum time to include in adjusted data

        Returns:
            DataFrame with adjusted timestamps and filtered time range
        """
        df_adjusted = dataframe.copy()

        # Adjust timestamps
        for time_col in ["timestamp", "time_since_zero"]:
            df_adjusted[time_col] -= start_delay

        # Filter time range
        df_filtered = df_adjusted[df_adjusted["time_since_zero"] >= 0]
        max_time = end_time - start_delay

        return df_filtered[df_filtered["time_since_zero"] <= max_time]

    def _resample_waveform(self, df: pd.DataFrame, key: str, timestamp: np.ndarray) -> pd.DataFrame:
        """Resample a waveform to a new timestamp array using linear interpolation"""
        return pd.DataFrame({"timestamp": timestamp, "value": np.interp(timestamp, df["time_since_zero"], df[key])})

    def align_data(self, motion_name, joint_name, data_type="positions"):
        """Align simulation and real data for comparison

        Args:
            motion_name: Name of the motion
            joint_name: Name of the joint
            data_type: Type of data to align (positions, velocities, torques)

        Returns:
            tuple: (aligned_sim, aligned_real) DataFrames with aligned timestamps
        """
        sim_df = self._motion_data[motion_name]["sim"].dof_state
        real_data = self._motion_data[motion_name]["real"]
        real_delay = real_data.real_event["MOTION_START"]
        real_end = real_data.real_event["DISABLE"]
        real_df = self.adjust_real_data_timing(real_data.dof_state, real_delay, real_end)

        key = f"{data_type}_{joint_name}"

        # Create uniform timestamp array for resampling
        max_time = min(sim_df["time_since_zero"].max(), real_df["time_since_zero"].max())
        timestamp = np.arange(self._sample_dt, max_time, self._sample_dt)

        # Resample both waveforms to the new timestamp
        aligned_sim = self._resample_waveform(sim_df, key, timestamp)
        aligned_real = self._resample_waveform(real_df, key, timestamp)

        return aligned_sim, aligned_real

    def generate_boxplots(self, output_dir):
        """
        Generate velocity and torque boxplots for all motions and joints.
        All joints for each motion and data type will be displayed in a single figure.

        Args:
            output_dir: Directory to save output files
        """
        # Only generate boxplots for velocities and torques
        data_types = ["velocities", "torques"]

        print(f"\n[PROCESSING] Generating boxplots for {len(self._motion_names)} motions...")

        for i, motion_name in enumerate(self._motion_names, 1):
            print(f"  [{i}/{len(self._motion_names)}] {motion_name}")

            # Create directory for saving images
            save_dir = Path(f"{output_dir}/{motion_name}/metrics_picture")
            save_dir.mkdir(parents=True, exist_ok=True)

            # Get joints for this motion
            joints = self._get_joints_for_motion(motion_name)
            num_joints = len(joints)

            if num_joints == 0:
                print(f"    [ERROR] No joints found for motion: {motion_name}")
                continue

            for data_type in data_types:
                try:
                    # Determine the optimal subplot layout based on the number of joints
                    if num_joints <= 2:
                        n_rows, n_cols = 1, 2
                    elif num_joints <= 4:
                        n_rows, n_cols = 2, 2
                    elif num_joints <= 6:
                        n_rows, n_cols = 2, 3
                    elif num_joints <= 9:
                        n_rows, n_cols = 3, 3
                    elif num_joints <= 12:
                        n_rows, n_cols = 3, 4
                    elif num_joints <= 16:
                        n_rows, n_cols = 4, 4
                    elif num_joints <= 25:
                        n_rows, n_cols = 5, 5
                    else:
                        # For large numbers of joints, use a 6x5 grid
                        n_rows, n_cols = 6, 5

                    # First collect all data
                    all_joint_data = []

                    for joint_name in joints:
                        try:
                            # Get aligned sim and real data
                            aligned_sim, aligned_real = self.align_data(motion_name, joint_name, data_type)

                            # Save data for later use
                            all_joint_data.append(
                                {
                                    "joint_name": joint_name,
                                    "sim_data": aligned_sim["value"],
                                    "real_data": aligned_real["value"],
                                    "error": None,
                                }
                            )
                        except Exception as e:
                            all_joint_data.append({"joint_name": joint_name, "error": str(e)})

                    # Create figure with subplots
                    fig_size = (n_cols * 4, n_rows * 3)  # Scale figure size based on subplot grid
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
                    fig.suptitle(f"{motion_name} - {data_type.capitalize()} Boxplots", fontsize=16)

                    # Make axes accessible for any subplot layout
                    if n_rows == 1 and n_cols == 1:
                        axes = np.array([[axes]])
                    elif n_rows == 1:
                        axes = axes.reshape(1, -1)
                    elif n_cols == 1:
                        axes = axes.reshape(-1, 1)

                    # First pass: Create all boxplots to get their actual display ranges
                    all_boxplots = []
                    for i, joint_data in enumerate(all_joint_data):
                        if i >= n_rows * n_cols:  # Skip if we have more joints than subplots
                            continue

                        row = i // n_cols
                        col = i % n_cols
                        ax = axes[row, col]

                        if "error" not in joint_data or not joint_data["error"]:
                            # Prepare data
                            data = [joint_data["sim_data"], joint_data["real_data"]]
                            # Create temporary boxplot to get statistics
                            bp = ax.boxplot(
                                data,
                                patch_artist=True,
                                labels=["Sim", "Real"],
                                showmeans=True,
                                meanline=False,
                                showfliers=False,
                            )
                            all_boxplots.append(
                                {"ax": ax, "boxplot": bp, "data": data, "joint_name": joint_data["joint_name"]}
                            )

                    # Get global y-limits from actual boxplot ranges
                    global_ymin = float("inf")
                    global_ymax = float("-inf")

                    for bp_data in all_boxplots:
                        bp = bp_data["boxplot"]
                        # Get min and max from whiskers
                        for whisker in bp["whiskers"]:
                            y_data = whisker.get_ydata()
                            global_ymin = min(global_ymin, np.min(y_data))
                            global_ymax = max(global_ymax, np.max(y_data))

                    # Add margin
                    if global_ymin != float("inf") and global_ymax != float("-inf"):
                        y_margin = 0.05 * (global_ymax - global_ymin)
                        global_ymin -= y_margin
                        global_ymax += y_margin
                    else:
                        # If no valid data, use default range
                        global_ymin, global_ymax = 0, 1
                        print(
                            f"    [WARNING] No valid data found for {motion_name} - {data_type}, "
                            f"using default y-limits"
                        )

                    # Clear all axes to redraw properly
                    for i in range(min(n_rows * n_cols, len(all_joint_data))):
                        row = i // n_cols
                        col = i % n_cols
                        axes[row, col].clear()

                    # Second pass: Redraw all boxplots with consistent styling and unified y-limits
                    for i, joint_data in enumerate(all_joint_data):
                        if i >= n_rows * n_cols:  # Skip if we have more joints than subplots
                            print(f"    [WARNING] Skipping joint {joint_data['joint_name']} (too many joints)")
                            continue

                        row = i // n_cols
                        col = i % n_cols
                        ax = axes[row, col]

                        if "error" in joint_data and joint_data["error"]:
                            ax.text(
                                0.5,
                                0.5,
                                f"Error: {joint_data['error'][:30]}...",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_title(joint_data["joint_name"], fontsize=10)
                        else:
                            # Prepare data
                            data = [joint_data["sim_data"], joint_data["real_data"]]
                            labels = ["Sim", "Real"]

                            # Create boxplot
                            boxplot = ax.boxplot(
                                data,
                                patch_artist=True,
                                labels=labels,
                                showmeans=True,
                                meanline=False,
                                showfliers=False,
                                meanprops={
                                    "marker": "o",
                                    "markerfacecolor": "white",
                                    "markeredgecolor": "blue",
                                    "markersize": 6,
                                },
                            )

                            # Set colors
                            colors = ["lightblue", "lightpink"]
                            for j, (patch, color) in enumerate(zip(boxplot["boxes"], colors)):
                                patch.set_facecolor(color)
                                # Add mean value text annotation
                                mean_val = np.mean(data[j])
                                ax.text(j + 1, mean_val, f"{mean_val:.2f}", ha="center", va="bottom", fontsize=8)

                            # Add title and labels
                            ax.set_title(joint_data["joint_name"], fontsize=10)
                            ax.grid(True, linestyle="--", alpha=0.7)

                        # Set unified y-axis range for all subplots
                        ax.set_ylim(global_ymin, global_ymax)

                    # Hide empty subplots
                    for i in range(len(all_joint_data), n_rows * n_cols):
                        row = i // n_cols
                        col = i % n_cols
                        fig.delaxes(axes[row, col])

                    # Add shared y-label
                    fig.text(
                        0.04, 0.5, f"{data_type.capitalize()} values", va="center", rotation="vertical", fontsize=12
                    )

                    # Adjust layout
                    plt.tight_layout(rect=[0.05, 0, 1, 0.95])  # Make room for y-label

                    # Save figure
                    plt.savefig(f"{save_dir}/{motion_name}_{data_type}_boxplot_all_joints.png", dpi=150)
                    plt.close(fig)

                    print(f"    [SUCCESS] Generated {data_type} boxplot ({num_joints} joints)")

                except Exception as e:
                    print(f"    [ERROR] Error generating {data_type} boxplot: {e}")

            print(f"    [SUCCESS] Completed boxplots for {motion_name}")

    def calculate_metrics(self, motion_name, joint_name, data_type="positions"):
        """Calculate comparison metrics for a single joint and data type

        Args:
            motion_name: Name of the motion
            joint_name: Name of the joint
            data_type: Type of data to analyze (positions, velocities, torques)

        Returns:
            dict: Dictionary of calculated metrics
        """
        # Align data
        aligned_sim, aligned_real = self.align_data(motion_name, joint_name, data_type)

        # Calculate metrics
        metrics = {}

        # 1. RMSE (Mean Squared Error)
        mse = np.mean((aligned_sim["value"] - aligned_real["value"]) ** 2)
        metrics["rmse"] = np.sqrt(mse)

        # 2. MAPE (Mean Absolute Percentage Error)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Filter out real values with absolute value less than 0.1
            valid_indices = np.abs(aligned_real["value"]) >= 0.1
            filtered_sim = aligned_sim[valid_indices]
            filtered_real = aligned_real[valid_indices]

            # Check if there are valid data points left
            if len(filtered_real) == 0:
                metrics["mape"] = np.nan
            else:
                abs_percentage_errors = np.abs(
                    (filtered_sim["value"] - filtered_real["value"]) / filtered_real["value"]
                )
                valid_errors = abs_percentage_errors[
                    ~np.isinf(abs_percentage_errors) & ~np.isnan(abs_percentage_errors)
                ]
                if len(valid_errors) > 0:
                    metrics["mape"] = np.mean(valid_errors) * 100  # as percentage
                else:
                    metrics["mape"] = np.nan

        # 3. Correlation
        correlation = np.corrcoef(aligned_sim["value"], aligned_real["value"])[0, 1]
        metrics["corr"] = correlation

        # 4. Cosine similarity
        similarity = 1 - distance.cosine(aligned_sim["value"], aligned_real["value"])
        metrics["cos_sim"] = similarity

        # 5. Cross-correlation for lag analysis
        corr = signal.correlate(aligned_sim["value"], aligned_real["value"], method="fft")
        # lags = signal.correlation_lags(len(aligned_sim["value"]), len(aligned_real["value"]))
        corr /= np.max(corr)

        max_corr_idx = np.argmax(np.abs(corr))
        max_lag = (max_corr_idx - (len(aligned_sim["value"]) - 1)) * self._sample_dt  # Convert to seconds
        metrics["max_lag"] = max_lag

        # 5. Smoothness metrics
        # Calculate derivatives
        if data_type != "positions":
            sim_diff = np.diff(aligned_sim["value"]) / self._sample_dt
            real_diff = np.diff(aligned_real["value"]) / self._sample_dt

            metrics["sim_smoothness_mean"] = np.mean(np.abs(sim_diff))
            metrics["sim_smoothness_var"] = np.var(sim_diff)
            metrics["real_smoothness_mean"] = np.mean(np.abs(real_diff))
            metrics["real_smoothness_var"] = np.var(real_diff)

        return metrics

    def analyze_all_data(self, output_dir, metrics_summary_file_name):
        """Analyze all motion data and generate Excel file with metrics

        Args:
            output_dir: Directory to save output files

        Returns:
            dict: Dictionary containing all calculated metrics
        """

        # Check if output_dir exists
        if os.path.exists(output_dir):
            # Check if it is a directory
            if not os.path.isdir(output_dir):
                raise ValueError(f"The provided output_dir '{output_dir}' exists but is not a directory.")
        else:
            # Create the directory if it does not exist
            os.makedirs(output_dir, exist_ok=True)

        self.generate_boxplots(output_dir)

        metrics_summary_filepath = os.path.join(output_dir, "metrics_summary.xlsx")

        # Set default file name if not provided
        if not metrics_summary_file_name:
            metrics_summary_file_name = "metrics_summary.xlsx"

        # Full path for the metrics summary file
        metrics_summary_filepath = os.path.join(output_dir, metrics_summary_file_name)

        # Define data types to analyze
        data_types = ["positions", "velocities", "torques"]

        # Define metrics to track and classify them
        comparison_metrics = ["rmse", "mape", "corr", "cos_sim", "max_lag"]

        sim_only_metrics = ["sim_smoothness_mean", "sim_smoothness_var"]
        real_only_metrics = ["real_smoothness_mean", "real_smoothness_var"]

        # Create dataframes to store results for each metric and data type
        metric_dfs = {}
        for data_type in data_types:
            metric_dfs[data_type] = {}

            # Add comparison metrics for all data types
            for metric in comparison_metrics:
                metric_dfs[data_type][metric] = pd.DataFrame()

            # Add smoothness metrics only for velocities and torques
            if data_type != "positions":
                for metric in sim_only_metrics + real_only_metrics:
                    metric_dfs[data_type][metric] = pd.DataFrame()

        # Process each motion and joint
        print(f"\n[PROCESSING] Calculating metrics for {len(self._motion_names)} motions...")

        for motion_idx, motion_name in enumerate(self._motion_names, 1):
            print(f"  [{motion_idx}/{len(self._motion_names)}] {motion_name}")

            all_joints = self._get_joints_for_motion(motion_name)
            print(f"    [INFO] Processing {len(all_joints)} joints × {len(data_types)} data types...")

            for joint_name in all_joints:
                # Calculate metrics for each data type
                for data_type in data_types:
                    try:
                        metrics = self.calculate_metrics(motion_name, joint_name, data_type)

                        # Add metrics to dataframes
                        for metric, value in metrics.items():
                            if joint_name not in metric_dfs[data_type][metric].index:
                                metric_dfs[data_type][metric].loc[joint_name, motion_name] = value
                            else:
                                metric_dfs[data_type][metric].at[joint_name, motion_name] = value
                    except Exception as e:
                        print(f"      [ERROR] {joint_name} ({data_type}): {str(e)[:50]}...")

        def format_excel_worksheet(
            writer, worksheet, dataframe, metric_name, sheet_name, index=True, header=True, extra_space=2
        ):
            """
            Format Excel worksheet with appropriate cell formats and column widths.

            Args:
                writer: Excel writer object
                worksheet: Excel worksheet object
                dataframe: Pandas DataFrame written to the worksheet
                metric_name: Name of the metric (e.g., 'mape', 'rmse', etc.)
                sheet_name: Name of the sheet
                index: Boolean indicating if index was included when writing DataFrame
                header: Boolean indicating if header was included when writing DataFrame
                extra_space: Additional space to add to column width
            """
            # Adjust column widths
            # Start from the first column (0 for index, 1 for data)
            start_col = 0

            # Adjust the index column width if index is included
            if index:
                idx_width = (
                    max(
                        len(str(dataframe.index.name or "")),  # Index name length
                        dataframe.index.astype(str).map(len).max(),  # Max index value length
                    )
                    + extra_space
                )
                worksheet.set_column(0, 0, idx_width)
                start_col = 1

            # Adjust each data column width
            for i, col in enumerate(dataframe.columns):
                col_width = (
                    max(
                        len(str(col)),  # Column name length
                        dataframe[col].astype(str).map(len).max() if len(dataframe) > 0 else 0,  # Max data length
                    )
                    + extra_space
                )
                worksheet.set_column(i + start_col, i + start_col, col_width)

            # Freeze top row and first column if both header and index are included
            if header and index:
                worksheet.freeze_panes(1, 1)
            elif header:
                worksheet.freeze_panes(1, 0)
            elif index:
                worksheet.freeze_panes(0, 1)

            # Set cell formats based on metric type
            workbook = writer.book

            # Create appropriate format based on metric
            if metric_name == "mape":
                # MAPE displayed as percentage with 4 decimal places
                cell_format = workbook.add_format({"num_format": "0.0000%"})
                # Apply percentage format (divide by 100 since MAPE is already multiplied by 100)
                for col_num in range(1, len(dataframe.columns) + 1):
                    for row_num in range(1, len(dataframe.index) + 1):
                        worksheet.write(row_num, col_num, dataframe.iloc[row_num - 1, col_num - 1] / 100, cell_format)
            else:
                # Other metrics with 4 decimal places
                cell_format = workbook.add_format({"num_format": "0.0000"})
                # Apply number format
                for col_num in range(1, len(dataframe.columns) + 1):
                    for row_num in range(1, len(dataframe.index) + 1):
                        worksheet.write(row_num, col_num, dataframe.iloc[row_num - 1, col_num - 1], cell_format)

        # Also create a combined Excel file with all data types
        with pd.ExcelWriter(
            metrics_summary_filepath, engine="xlsxwriter", engine_kwargs={"options": {"nan_inf_to_errors": True}}
        ) as writer:
            # Add sheets for each data type and metric
            for data_type in data_types:
                # Add comparison metrics with _sim_real suffix
                for metric in comparison_metrics:
                    sheet_name = f"{data_type}_{metric}_compare"
                    metric_dfs[data_type][metric].to_excel(writer, sheet_name=sheet_name)
                    format_excel_worksheet(
                        writer, writer.sheets[sheet_name], metric_dfs[data_type][metric], metric, sheet_name
                    )

                if data_type != "positions":
                    # Add sim-only metrics with _sim suffix
                    for metric in sim_only_metrics:
                        sheet_name = f"{data_type}_{metric.replace('sim_', '')}_sim"
                        metric_dfs[data_type][metric].to_excel(writer, sheet_name=sheet_name)
                        format_excel_worksheet(
                            writer, writer.sheets[sheet_name], metric_dfs[data_type][metric], metric, sheet_name
                        )

                    # Add real-only metrics with _real suffix
                    for metric in real_only_metrics:
                        sheet_name = f"{data_type}_{metric.replace('real_', '')}_real"
                        metric_dfs[data_type][metric].to_excel(writer, sheet_name=sheet_name)
                        format_excel_worksheet(
                            writer, writer.sheets[sheet_name], metric_dfs[data_type][metric], metric, sheet_name
                        )

        print(f"\n[SAVED] Combined metrics to: {metrics_summary_filepath}")
        print("[SUCCESS] Analysis completed successfully!")

        return metric_dfs

    def plot_comparison_data(self, axes, sim_cmd, sim_state, real_cmd, real_state, joint_name, plot_titles):
        """Plot comparison data for simulation and real robot

        Args:
            axes: List of matplotlib axes for plotting
            sim_cmd: Simulation command data
            sim_state: Simulation state data
            real_cmd: Real robot command data
            real_state: Real robot state data
            joint_name: Name of the joint being plotted
            plot_titles: Titles for each subplot
        """
        # Plot command position and position on the same graph
        axes[0].plot(
            sim_cmd["time_since_zero"], sim_cmd[f"positions_{joint_name}"], "#add8e6", label="Sim Command Position"
        )  # Light blue for sim command positions
        axes[0].plot(
            real_cmd["time_since_zero"], real_cmd[f"positions_{joint_name}"], "#ffcccb", label="Real Command Position"
        )  # Light red for real command positions
        axes[0].plot(
            sim_state["time_since_zero"], sim_state[f"positions_{joint_name}"], "b--", label="Sim Position"
        )  # Dashed blue for sim positions
        axes[0].plot(
            real_state["time_since_zero"], real_state[f"positions_{joint_name}"], "r--", label="Real Position"
        )  # Dashed red for real positions
        axes[0].set_title(plot_titles[0])
        axes[0].legend()
        axes[0].grid(True)

        # Plot velocity
        axes[1].plot(sim_state["time_since_zero"], sim_state[f"velocities_{joint_name}"], "b--", label="Sim Velocity")
        axes[1].plot(
            real_state["time_since_zero"], real_state[f"velocities_{joint_name}"], "r--", label="Real Velocity"
        )
        axes[1].set_title(plot_titles[1])
        axes[1].legend()
        axes[1].grid(True)

        # Plot torque
        axes[2].plot(sim_state["time_since_zero"], sim_state[f"torques_{joint_name}"], "b--", label="Sim Torque")
        axes[2].plot(real_state["time_since_zero"], real_state[f"torques_{joint_name}"], "r--", label="Real Torque")
        axes[2].set_title(plot_titles[2])
        axes[2].legend()
        axes[2].grid(True)

    def visualize_joint_comparison(self, motion_name, joint_name, output_dir, save_plot=True, show_plot=False):
        """Visualize comparison between simulation and real robot data for a specific joint

        Args:
            motion_name: Name of the motion to visualize
            joint_name: Name of the joint to visualize
            output_dir: Directory to save output files
            save_plot: Whether to save the plot to file
            show_plot: Whether to display the plot
        """
        if motion_name not in self._motion_data:
            print(f"[ERROR] Motion {motion_name} not found in loaded data.")
            return

        sim_data = self._motion_data[motion_name]["sim"]
        real_data = self._motion_data[motion_name]["real"]

        # Create save directory
        save_dir = Path(f"{output_dir}/metrics/{self._robot_name}/{self._motion_source}/{motion_name}/")
        if save_plot:
            save_dir.mkdir(parents=True, exist_ok=True)

        # Removed redundant processing logs

        # Create figure and subplots
        fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharex=False)
        plot_titles = ["Position (rad)", "Velocity (rad/s)", "Torque (Nm)"]

        # Get data
        sim_cmd = sim_data.dof_command
        sim_state = sim_data.dof_state

        # Adjust real data timing
        real_start = real_data.real_event["MOTION_START"]
        real_end = real_data.real_event["DISABLE"]
        real_cmd = self.adjust_real_data_timing(real_data.dof_command, real_start, real_end)
        real_state = self.adjust_real_data_timing(real_data.dof_state, real_start, real_end)

        # Plot comparison data
        self.plot_comparison_data(axes, sim_cmd, sim_state, real_cmd, real_state, joint_name, plot_titles)

        # Dynamically adjust y-axis limits for each subplot
        for ax in axes:
            y_min, y_max = ax.get_ylim()
            abs_max = max(abs(y_min), abs(y_max))
            if abs_max < 0.1:
                ax.set_ylim(-0.1, 0.1)  # Set y-axis limit to [-0.1, 0.1]

        # Set figure title and labels
        fig.suptitle(f"Comparison for {joint_name} - Motion: {motion_name}")
        fig.supxlabel("Time Since Zero (s)")
        plt.tight_layout()

        # Save or display plot
        if save_plot:
            plt.savefig(f"{save_dir}/{joint_name}.png")
            # Results saved quietly - detailed logs removed for cleaner output
        if show_plot:
            plt.show()

        # Close the figure
        plt.close(fig)

    def visualize_all_comparisons(self, output_dir, save_plot=True, show_plot=False):
        """Visualize comparisons for all motions and joints

        Args:
            output_dir: Directory to save output files
            save_plot: Whether to save the plot to file
            show_plot: Whether to display the plot
        """
        print("\n[PROCESSING] Generating visualization plots...")

        total_plots = sum(len(self._get_joints_for_motion(motion)) for motion in self._motion_data.keys())
        plot_count = 0

        for motion_name, data in self._motion_data.items():
            joints = self._get_joints_for_motion(motion_name)
            print(f"  [INFO] {motion_name}: {len(joints)} joint plots")
            for joint_name in joints:
                plot_count += 1
                if plot_count % max(1, total_plots // 10) == 0:  # Show progress every 10%
                    print(f"    [PROGRESS] {plot_count}/{total_plots} plots completed")
                self.visualize_joint_comparison(motion_name, joint_name, output_dir, save_plot, show_plot)

        print(f"[SUCCESS] Generated {total_plots} visualization plots")
