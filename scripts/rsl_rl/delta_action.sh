#!/bin/bash

letter="a_0" 

python scripts/rsl_rl/train.py --task Isaac-Humanoid-Motor-Direct-v0 --letter "$letter" --headless
python scripts/rsl_rl/play.py --task Isaac-Humanoid-Motor-Direct-v0 --num_envs 1 --letter "$letter" --headless
python scripts/rsl_rl/plot_delta_action.py --letter "$letter" --frequency 50 --motion_num 6