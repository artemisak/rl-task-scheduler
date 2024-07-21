# RL Task Scheduler

This project implements a multi-agent reinforcement learning system for optimizing surgery scheduling in a hospital environment. It uses the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm to train agents to make efficient scheduling decisions.

# Installation and run
There is a fast way to install all dependencies and run the experiment using bash script `run.sh`. First you need to open a project folder in the Visual Studio Code and call the terminal. Write this line in a bash console:

```
./run.sh
```
This will start the next pipeline: learning process, evaluation and visualization. It's recommended to set up smoothing to 0.85 for plots are highly oscillating.

Note, in some cases you need to give your system a permission to run this script:
```
chmod + run.sh
```

If you wish to install and launch all by yourself, there is a `requirements.txt` file. In this case open a terminal window and write next to create a separated copy of a virtual environment:

```
python3 -m venv env
```
Than you need to get to a `env/bin` or `env/Scripts` directory which depences on your system and activate the virtual environment with next command:

```
source activate
```

Probably now you will see a `(env)` apostroph in you terminal. When ready write next to install all the necessary requirements:

```
python3 -m pip install -r requirements.txt
```
None, in some cases you need to change `python3` to `python` or whatever alias is already set in your system.

Now you will be able to separatly launch `train.py` and `evaluate.py` python scripts like this:

```
python3 train.py
```
