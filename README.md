# RL Task Scheduler

## Overview

This project implements a multi-agent reinforcement learning system for optimizing surgery scheduling in a hospital environment. It uses the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm to train agents to make efficient scheduling decisions.

## Files

### env.py

This file contains the implementation of the `SurgeryQuotaScheduler` environment. It simulates a hospital scheduling system where multiple agents (representing surgeries) need to be allocated across a 14-day period. The environment handles the dynamics of surgery urgency, complexity, and scheduling conflicts.

Key features:
- Custom ParallelEnv implementation
- Dynamic surgery attributes (urgency, completeness, complexity)
- Reward function based on scheduling efficiency
- Visualization of the scheduling state (when render mode is 'human')

### maddpg_run.py

This script implements the training loop for the MADDPG algorithm on the Surgery Quota Scheduler environment. It utilizes the AgileRL library for reinforcement learning components and implements additional features for performance tracking and optimization.

Key features:
- MADDPG algorithm implementation
- Population-based training with evolutionary hyperparameter optimization
- TensorBoard integration for performance logging
- Customizable network architecture and hyperparameters

### run.sh

This bash script automates the setup and execution of the training process. It handles environment setup, dependency installation, and runs the main training script.

## Dependencies

- Python 3.7+
- PyTorch
- NumPy
- Pygame (for visualization)
- AgileRL
- TensorBoard

## Usage

1. Ensure you have bash installed on your system.
2. Make the run script executable:

```bash
chmod +x run.sh
```

3. Execute the run script:

```bash
./run.sh
```

This script will set up the necessary environment, install dependencies, and start the training process. The script will output progress information and save the trained model upon completion.

## Note

Detailed documentation, including API references and in-depth explanations of the algorithm and environment, will be provided in a separate DOCUMENTATION file.
