# Installation
```bash
conda create -n realman python=3.9
conda activate realman
pip install numpy matplotlib h5py joblib scipy

python ui.py
```


# Run
`ui.py` used to collect all the data. You can choice motion index in ui.

`collect.py` defined the function to collect single motion data.

The `realman_class.py` defined the class connect realman robot (Dual and Humanoid).

Run `data_record_humanoid.py` to run the robot.

You need to save the motion file(.pickle) in `/data`.

The results will save in `/data/runs` as a `.h5` file. You can use `plot_from_hdf5.py` to show the curve. The plots will save at `plot_data`.





