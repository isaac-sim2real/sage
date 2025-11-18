# Q&A - Frequently Asked Questions

This document records common issues and solutions encountered when using the SAGE project, helping users quickly resolve problems during code reproduction.

## Table of Contents

- [Configuration](#configuration)
- [Simulation](#simulation)
- [Real Robot Integration](#real-robot-integration)
- [Other Issues](#other-issues)

---

## Configuration

### Q1: Must the kp/kd parameters used during data collection be the same as those used during policy deployment?

**A1:** Yes, they must be consistent. The kp/kd parameters used for data collection (both in simulation and on real robots) should match those used during policy deployment to ensure accurate sim2real gap analysis and model training.

### Q2: Where can I configure different kp/kd values for different joints and load joint inertia information?

**A2:** Robot configuration parameters (including kp/kd values and joint properties) are centralized in [`sage/assets.py`](https://github.com/j3soon/isaac-sim2real-sage/blob/main/sage/assets.py). You can also override these parameters using command-line arguments when running scripts. For more details, see the [Robot Configuration Parameters](https://github.com/j3soon/isaac-sim2real-sage/tree/main?tab=readme-ov-file#robot-configuration-parameters) section in the README.

---

## Simulation



## Real Robot Integration

### Q1: Why attach weights to real robots instead of using domain randomization in simulation and treating it as real data?

**A1:** As payload increases, real robot performance diverges from simulation - real robots perform worse than simulated ones. Using standardized weights (e.g., 0kg, 1kg, 2kg, 3kg) provides benchmark data for more accurate sim-to-real comparison rather than relying purely on randomized simulation data.

---

## Other Issues

### Q1: Where is the training code for the gap compensation model (Gaponet)?

**A1:** The SAGE project currently includes data collection and analysis components only. The training code for the gap compensation model is hosted at Peking University's lab and will be open-sourced after the related paper is published.

---

**Last Updated:** 2025-11-18

