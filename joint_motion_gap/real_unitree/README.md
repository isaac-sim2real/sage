# Guidance for Data Collection on Unitree Robot G1 & H12

## Package Installation
Refer to [Unitree_ros2](https://github.com/unitreerobotics/unitree_ros2), we need install both the Unitree sdk and ros_foxy

(Maybe we can add more details about installation of ROS2 and SDK to fill our Document)

## Data Collection

A unified Python script manages data collection. The target robot for control is selected via the `--robot` command-line interface (CLI) argument.


### H1-2
```
python hardware_data_collect.py --robot h12
```

### G1
```
python hardware_data_collect.py --robot g1
```

## Output

The data will be saved in data/runs_csv as the format of simulation

