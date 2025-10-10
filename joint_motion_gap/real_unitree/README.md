**TODO:**
- [ ] **Installation**需要add more details about installation of ROS2 and SDK to fill our Document
- [ ] **3. Data Management**和**4. Data Visualization**需要同realman一样进行补充。
- [ ] 最好补充一个**Directory Structure**部分

# Unitree Robot G1 & H12 Data Collection

This project provides tools for collecting and processing motion data from the Unitree robot.


## Installation

Refer to [Unitree_ros2](https://github.com/unitreerobotics/unitree_ros2), we need install both the Unitree sdk and ros_foxy


## Usage

### 1. Data Collection

A unified Python script manages data collection. The target robot can be specified via the `--robot` command-line interface (CLI) argument.

- Collecting Data from G1
    ```bash
    python hardware_data_collect.py --robot g1
    ```

- Collecting Data from H1-2
    ```bash
    python hardware_data_collect.py --robot h12
    ```

### 2. Output

Collected data will be saved in the `data/runs_csv` directory in a simulation-compatible format.

### 3. Data Management


### 4. Data Visualization


## Directory Structure







