**TODO:**
- [ ] 没有**1. Data Collection**的**ui.py**和**collect.py**文件，需要修改
- [ ] **1. Data Collection**和**2. Robot Control**分别有record和collect部分，究竟是什么顺序？不太看得懂。
- [ ] 最好补充一个**Directory Structure**部分

# Realman Robot Data Collection

This project provides tools for collecting and processing motion data from the Realman robot.


## Installation

1. **Create and activate a Conda environment:**
    ```bash
    conda create -n realman python=3.9
    conda activate realman
    ```
2. **Install required packages:**
    ```bash
    pip install numpy matplotlib h5py joblib scipy
    ```


## Usage

### 1. Data Collection

- Run the UI to start collecting data. You can select the motion index in the interface:
    ```bash
    python ui.py
    ```
- Core data collection functions are defined in `collect.py`.

### 2. Robot Control

- The `realman_class.py` file defines classes for connecting and controlling the Realman robot (both Dual and Humanoid).
- Use `data_record_humanoid.py` to run the robot and record motion data.

### 3. Data Management

- Save your motion files (with `.pickle` extension) in the `/data` directory.
- Run results will be saved as `.h5` files in `/data/runs`.

### 4. Data Visualization

- Use `plot_from_hdf5.py` to visualize results from `.h5` files:
    ```bash
    python plot_from_hdf5.py
    ```
- Generated plots will be saved in the `plot_data` directory.


## Directory Structure





