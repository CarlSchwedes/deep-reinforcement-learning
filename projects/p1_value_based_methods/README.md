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

1. Create and activate a Python virtual environment.
2. Install package dependencies from the Unity ML-Agents Python package in this repository.
3. Download and unzip the correct Unity environment executable for your OS.
4. Verify notebook kernel points to your active environment.

Example setup commands:

- Windows

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
cd python
pip install .
cd ..
```

- Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
cd python
pip install .
cd ..
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

Vector-state training (required project path):
1. Open p1_navigation/Navigation.ipynb
2. Set the correct Unity executable path in the environment initialization cell.
3. Run notebook cells from top to bottom.
4. Training saves checkpoint.pth when solve criterion is reached.

Pixel-based training (optional challenge path):
1. Open p1_navigation/Navigation_Pixels.ipynb
2. Set USE_VISUAL_FEATURES = True for CNN mode, or False to use vector observations with MLP.
3. Set the correct VisualBanana executable path.
4. Run notebook cells from top to bottom.

Main implementation files used during training:
- p1_navigation/dqn_agent.py
- p1_navigation/model.py

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

1. Create (and activate) a virtual environment named `.venv` with Python 3.13.7 (tested).

	- __Linux__ or __Mac__: 
	```bash
	python3.13 -m venv .venv
	source .venv/bin/activate
	python --version  # should show Python 3.13.7
	```
	- __Windows__: 
	```bash
	py -3.13 -m venv .venv
	.venv\Scripts\activate
	python --version  # should show Python 3.13.7
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/Value-based-methods.git
cd Value-based-methods/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `.venv` environment.  
```bash
python -m ipykernel install --user --name .venv --display-name "Python (.venv)"
```

5. Before running code in a notebook, change the kernel to match the `.venv` environment by using the drop-down `Kernel` menu. 

`drlnd` renamed -> `.venv`

![Kernel][image2]

## Want to learn more?

<p align="center">Come learn with us in the <a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">Deep Reinforcement Learning Nanodegree</a> program at Udacity!</p>

<p align="center"><a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">
 <img width="503" height="133" src="https://user-images.githubusercontent.com/10624937/42135812-1829637e-7d16-11e8-9aa1-88056f23f51e.png"></a>
</p>
