# Sim2Real delta action code base

After installed IsaacLab:

```bash
python -m pip install -e source/sim2real source/sim2real_assets
```

## WorkSpace Configuration

The workspace configuration file for this project is: `Sim2Real/.vscode/sim2real_codebase.code-workspace`. To ensure correct path indexing and development environment setup, you need to replace the "path" value with the actual path to IsaacLab on your local system:

```json
"name": "IsaacLab",
"path": "../../../projects/IsaacLab"
```

### train

```bash
cd Sim2Real
python scripts/rsl_rl/train.py --task Isaac-Humanoid-Motor-Direct-v0 --letter a_0 --headless
python scripts/rsl_rl/train.py --task Isaac-Humanoid-Amass_Delta_Action --letter amass --headless
```

### play

```bash
cd Sim2Real
python scripts/rsl_rl/play.py --task Isaac-Humanoid-Motor-Direct-v0 --num_envs 1 --letter a_0 --headless
python scripts/rsl_rl/play.py --task Isaac-Humanoid-Amass_Delta_Action --num_envs 1 --letter amass --headless
```

### run all

```bash
conda activate isaaclab
sh scripts/rsl_rl/delta_action.sh 
```

### Parameter Description

- `letter`：The first letter represents joint.
- `motion_num`：The number of subgraphs.
