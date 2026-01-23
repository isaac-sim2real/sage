# LeRobot SO-101 Real Robot Motion Data Collection

This project provides tools for collecting and processing motion data from LeRobot SO-101 follower arm. It supports loading motion sequences from TXT files and collecting real-time robot state data during motion execution.

## Installation

You can set up the real robot control environment either directly on your host machine or using Docker.

For host installation, follow [the official guide](https://huggingface.co/docs/lerobot/installation):

1. **Create and activate a Conda environment:**
    ```bash
    conda create -y -n lerobot python=3.10
    conda activate lerobot
    ```
2. **Install required packages:**
    ```bash
    conda install ffmpeg -c conda-forge  # optional
    pip install 'lerobot[feetech]'
    ```

For Docker installation, build the Docker image:

```sh
docker build -f Dockerfile_so101 -t sage:so101 .
```

Then run the container:

```sh
docker run -it --rm --privileged \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v /dev:/dev \
    -v $(pwd):/app \
    sage:so101
```

## Usage

### Execute Motion and Collect Data

```bash
sudo chmod 666 /dev/ttyACM*  # Skip this step if running in Docker container as root
# Modify the optional arguments as needed
python scripts/run_real.py \
    --robot-name=so101 \
    --motion-files=custom/custom_motion.txt \
    --output-folder=output \
    --robot-port=/dev/ttyACM0 \
    --robot-type=so101_follower \
    --robot-id=my_awesome_follower_arm
```

## Input Format

### TXT Motion Files
Motion files should be in CSV format:
```
Rotation,Pitch,Elbow,Wrist_Pitch,Wrist_Roll,Jaw
0.1,0.2,0.3,0.4,0.5,0.6
...
```

- **First row**: Joint names (6 joints)
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
