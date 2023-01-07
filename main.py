from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import torch

from self_driving_car_env.env import SelfDrivingCar

# from params import args
from params import args
print(args)
# training dqn agent
def train_dqn(n_episodes= 200, max_t = 1500, eps_start=1.0, eps_end = 0.01,
       eps_decay=0.996):
    """Deep Q-Learning
    
    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon 
        eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon
        
    """
    scores = [] # list containing score from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state,eps)
            next_state,reward,done,_ = env.step(action)
            agent.step(state,action,reward,next_state,done)
            ## above step decides whether we will train(learn) the network
            ## actor (local_qnetwork) or we will fill the replay buffer
            ## if len replay buffer is equal to the batch size then we will
            ## train the network or otherwise we will add experience tuple in our 
            ## replay buffer.

            state = next_state
            score += reward
            if done:
                break
            scores_window.append(score) ## save the most recent score
            scores.append(score) ## sae the most recent score
            eps = max(eps*eps_decay,eps_end)## decrease the epsilon
            # print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)), end="")
        
        if i_episode % 50==0:
            print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))
            
        if np.mean(scores_window)>=2000.0 or max_t == t:
            print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                        np.mean(scores_window)))
            # save the model to the file path
            torch.save(agent.qnetwork_local.state_dict(),args["save_model_path"]+'checkpoint.pth')
            break

    return scores

# create an agent
agent = DQNAgent(state_size=5,
                action_size=3,
                seed=0,
                buffer_size = args["buffer_size"],
                batch_size = args["batch_size"],
                gamma = args["gamma"],
                tau = args["tau"],
                lr = args["lr"],
                update_every = args["update_every"])

if args["train"]:
    env = SelfDrivingCar(render_mode=args["render"])
    scores = train_dqn(n_episodes=args["n_episodes"],
                max_t = args["max_t"],
                eps_decay=args["eps_decay"],
                eps_start=args["eps_start"],
                eps_end=args["eps_end"])
else:
    # load the weights from file
    env = SelfDrivingCar(render_mode=True)
    print("Loading the model from the file path: ",args["load_model_path"])
    agent.load_model(args["load_model_path"])
    # play the game
    state = env.reset()
    score = 0
    while True:
        action = agent.act(state)
        next_state,reward,done,_ = env.step(action)
        state = next_state
        score += reward
        if done:
            env.reset()