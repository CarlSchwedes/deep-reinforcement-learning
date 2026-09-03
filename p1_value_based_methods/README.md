[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif "Trained Agents"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Value-Based Methods

![Trained Agents][image1]

This repository contains material related to Udacity's Value-based Methods course.

## Project 1 Submission Guide (Navigation)

This section is included to satisfy the Project 1 (Navigation) submission rubric.

### Project Details

- Environment: Unity Banana Navigation (single agent, discrete action space)
- State space (project environment): 37-dimensional vector observation
- Actions:
	- 0: move forward
	- 1: move backward
	- 2: turn left
	- 3: turn right
- Reward design:
	- +1 for yellow banana
	- -1 for blue banana
- Solved condition: average score >= +13 over 100 consecutive episodes

Optional challenge environment in this repository:
- VisualBanana pixel observations with raw image states (84 x 84 x 3 per frame)
- CNN agent consumes stacked grayscale frames (4 x 84 x 84)

### Getting Started

1. Create and activate a Conda environment using the provided environment.yml.
2. Download and unzip the correct Unity environment executable for your OS.
3. Verify notebook kernel points to your active environment.

Example setup commands:

- Windows

```bash
conda env create -f environment.yml
conda activate drl_env_py_3_9
python --version  # should show Python 3.9.23
```

- Linux or macOS

```bash
conda env create -f environment.yml
conda activate drl_env_py_3_9
python --version  # should show Python 3.9.23
```

Run tensorboard to visualize training metrics: (open new terminal, navigate to project root, and run)

```bash
tensorboard --logdir runs
```

Unity environment binaries (download one):
- Project (vector-state) Banana:
	- Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
	- Mac: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
	- Windows x86: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip
	- Windows x86_64: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip
- Optional pixel challenge VisualBanana:
	- Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip
	- Mac: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip
	- Windows x86: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip
	- Windows x86_64: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip

Place extracted folders under p1_navigation.

### Instructions to Train the Agent

**Primary Training Method: Python Scripts**

The main training implementation uses dedicated Python scripts for reproducible, scalable training runs:

Vector-state training (required project path):
1. Navigate to `navigation_vector/` directory
2. Edit `main.py` to set the correct Banana environment executable path
3. Run: `python main.py`
4. Training saves checkpoint.pth when solve criterion is reached

Pixel-based training (optional challenge path):
1. Navigate to `navigation_pixel/` directory
2. Edit `main.py` to set the correct VisualBanana environment executable path
3. Run: `python main.py`
4. Training saves checkpoint.pth when solve criterion is reached

**Notebooks for Initial Exploration (Ramp-up Phase)**

Jupyter notebooks are available for interactive learning and experimentation:
- `navigation_vector/Navigation.ipynb` – Vector-state agent development walkthrough
- `navigation_pixel/Navigation_Pixels.ipynb` – Pixel-state agent development walkthrough

These notebooks provide a guided learning experience before transitioning to the Python scripts.

**Main Implementation Files**

Core implementation used in both scripts and notebooks:
- `dqn_agent.py` – DQN agent with Rainbow components (PER, Double DQN, Dueling, Noisy Networks, N-step returns, Distributional RL)
- `model.py` – Neural network architectures (QNetwork, DuelingQNetwork, CNN variants, NoisyLinear layers)

Report plots are placed in `reports/` directory.

### Report

Project report is provided at repository root:
- REPORT.md

## Table of Contents

### Tutorials

The tutorials lead you through implementing various algorithms in reinforcement learning.  All of the code is in PyTorch (v0.4) and Python 3.

* [Deep Q-Network](https://github.com/udacity/Value-based-methods/tree/main/dqn): Explore how to use a Deep Q-Network (DQN) to navigate a space vehicle without crashing.

### Labs / Projects

The labs and projects can be found below.  All of the projects use rich simulation environments from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents).

* [Navigation](https://github.com/udacity/Value-based-methods/tree/main/p1_navigation): In the first project, you will train an agent to collect yellow bananas while avoiding blue bananas.

### Resources

* [Cheatsheet](https://github.com/udacity/Value-based-methods/tree/main/cheatsheet): You are encouraged to use [this PDF file](https://github.com/udacity/Value-based-methods/blob/main/cheatsheet/cheatsheet.pdf) to guide your study of reinforcement learning. 

## OpenAI Gym Benchmarks

### Box2d
- `LunarLander-v2` with [Deep Q-Networks (DQN)](https://github.com/udacity/Value-based-methods/blob/main/dqn/solution/Deep_Q_Network_Solution.ipynb) | solved in 1504 episodes

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create and activate the Conda environment from the project root `environment.yml` file.

	- __Linux__, __Mac__, or __Windows__: 
	```bash
	conda env create -f environment.yml
	conda activate drl_env_py_3_9
	python --version  # should show Python 3.9.23
	```

The `environment.yml` file includes all dependencies for running DRL agents and Unity ML-Agents, including PyTorch and Jupyter.
	
2. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drl_env_py_3_9` environment.  
```bash
python -m ipykernel install --user --name drl_env_py_3_9 --display-name "Python (drl_env_py_3_9)"
```

3. Before running code in a notebook, change the kernel to match the `drl_env_py_3_9` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

![Kernel][image2]

## Want to learn more?

<p align="center">Come learn with us in the <a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">Deep Reinforcement Learning Nanodegree</a> program at Udacity!</p>

<p align="center"><a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">
 <img width="503" height="133" src="https://user-images.githubusercontent.com/10624937/42135812-1829637e-7d16-11e8-9aa1-88056f23f51e.png"></a>
</p>
