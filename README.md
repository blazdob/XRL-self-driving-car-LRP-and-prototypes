# Explaining autonomous driving using two examples

This repository contains the code for the examples in the report for an PhD research done during an AI course at the university of Ljubljana. In order to access the final report you can access it via `results/explainable_reinforcement_learning_LRP_prototypes.pdf`.

## Installation

The code is written in Python 3.9. In order to reproduce the results, you need to install an virtual enviorment defined in the `requirements.yml` file. This can be done by running the following command:

```bash
conda env create -f environment.yml
```

## Contence of the repository

The code contains an environment for self driving car located in folder `self_driving_car_env`. A classical reinforcement learning agent is implemented using classic DQN algorithm and an agent that does the inicialisations of the networks and the actual learning of them as well. The agent also takes care of the experience replay and batches, numer of upades, learning rate, tau. etc...

The agent definition is located in `dqn_agent.py` and the plain learning without explanations and the default DQN agent is located in `main.py`. parameters for the agent are located in `params.py`. and can be changed there.

The code conatins two explanations examples:
* `main_lrp.py` - this file contains the code for the first example, where we explain the behaviour of the agent in the environment usin an alteration of LRP feature attribution to the individual sensors. The torch implementation of LRP layers is the work by Frederik Hvilshøj [https://github.com/fhvilshoj/TorchLRP].
* `main_prototypes.py` - this file contains the code for the second example, where we explain the behaviour of the agent in the environment using human defined prototypes of the environment.

## Running the code

In order to run the code, you need to activate the virtual enviorment and then run the code. The code can be run by running the following command:

```bash
conda activate self_driving_car
python main.py
```

### running the LRP example

In order to run the LRP example, you need to run the following command:

```bash
conda activate self_driving_car
python main_lrp.py
```

If you want to run a different rule for the LRP, you need to change the `rule` parameter in the `params.py` file.

### running the prototypes example

In order to run the prototypes example, you need to run the following command:

```bash
conda activate self_driving_car
python main_prototypes.py
```

## Results

The results of the code are located in the `results` folder. The results of the LRP example are located in the `results/lrp` folder and the results of the prototypes example are located in the `results/prototypes` folder.

The video of the Prototypes explanations is available on [youtube](https://youtu.be/cItgedx6mqw).

## Authors

* **Blaž Dobravec** - *Implementation and design work*
* **Jure Žabkar** - *Supervision - mentorship*
* **Ivan Bratko** - *Supervision*


