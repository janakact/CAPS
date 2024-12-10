# Constraint-Adaptive Policy Switching (CAPS) for Offline Safe Reinforcement Learning

This repository contains the implementation of Constraint-Adaptive Policy Switching (CAPS) for Offline Safe Reinforcement Learning (RL) AAAI 2025. The structure and setup of this repository follow closely with the [OSRL repository](https://github.com/liuzuxin/OSRL).

## Setup Instructions

1. **Environment Setup:**
   - Please follow the [OSRL installation guide](https://github.com/liuzuxin/OSRL) to set up the necessary environments and install all required dependencies.

2. **Integrating CAPS with OSRL:**
   - Copy the contents of the `examples/configs/` directory from this repository into `OSRL/examples/configs/`.
   - Copy the contents of the `examples/eval/` directory from this repository into `OSRL/examples/eval/`.
   - Copy the contents of the `examples/train/` directory from this repository into `OSRL/examples/train/`.
   - Copy the contents of the `osrl/algorithms/` directory from this repository into `OSRL/osrl/algorithms/`.
   - Append the contents of `osrl/common/net.py` from this repository to `OSRL/osrl/common/net.py`.


### Training

To train the `CAPS (IQL)` instantiation, override the default parameters as needed and run the following command:

```bash
python examples/train/train_capsiql.py --task OfflineCarCircle-v0 --param1 args1 ...
```

- The configuration file and training logs will be saved in the `logs_{num_heads}/` directory by default.
- Training plots can be monitored online via [Weights & Biases (Wandb)](https://wandb.ai/).

You can also execute a sequence of experiments or run them in parallel using the [EasyRunner](https://github.com/liuzuxin/easy-runner) package. See the `examples/train_all_tasks.py` script for more details.

### Evaluation

To evaluate a trained agent, such as a `CAPS IQL` agent, use the following command:

```bash
python examples/eval/eval_capsiql.py --path path_to_model --cost_limit 20 --eval_episodes 20
```

- This command will load the configuration from `path_to_model/config.yaml` and the model from `path_to_model/checkpoints/model.pt`.
- The agent will run for 20 episodes, and the average normalized reward and cost will be printed.

## Acknowledgements

The design of caps the methods and the implementation of baselines follow the OSRL repository.

