import os
import time
import torch
import pygame
import vidmaker
import numpy as np
from collections import deque

from dqn_agent import DQNAgent
from self_driving_car_env.env import SelfDrivingCar
from prototypes.prototype_tracker import PrototypeTracker
from interestingness.analyzer import Analyzer

# from params import args
from params import args

# training dqn agent
def train_prototypes_dqn(n_episodes= 200, max_t = 700, eps_start=1.0, eps_end = 0.01,
       eps_decay=0.996, save_behaviour=False):
    """Deep Q-Learning
    
    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon 
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        
    """
    scores = [] # list containing score from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    eps = eps_start

    prototype_tracker = PrototypeTracker(n_episodes)

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        rolling_window = 20
        for t in range(max_t):
            action = agent.act(state,eps)
            next_state,reward,done,_ = env.step(action)
            agent.step(state,action,reward,next_state,done)
            
            prototype_tracker.add_sample(state, action)
            ## above step decides whether we will train(learn) the network
            ## actor (local_qnetwork) or we will fill the replay buffer
            ## if len replay buffer is equal to the batch size then we will
            ## train the network or otherwise we will add experience tuple in our 
            ## replay buffer.
            meta_action = prototype_tracker.current_prototype_action()
            
            env.plot_meta(meta_action)
            state = next_state
            score += reward
            if done:
                break
            scores_window.append(score) ## save the most recent score
            scores.append(score) ## sae the most recent score
            eps = max(eps*eps_decay,eps_end)## decrease the epsilon
            # print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)), end="")
        else:
            print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                        np.mean(scores_window)))
            # save the model to the file path
            torch.save(agent.qnetwork_local.state_dict(),args["save_model_path"]+ 'model_pads1.pth')
            break

        prototype_tracker.new_episode()
        
        if i_episode % 50==0:
            print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))
            
        if np.mean(scores_window)>=2000.0 or max_t == t:
            print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                        np.mean(scores_window)))
            # save the model to the file path
            torch.save(agent.qnetwork_local.state_dict(),args["save_model_path"]+ 'model_pads1.pth')
            break
    # save the data from behavior tracker
    if save_behaviour:
        prototype_tracker.save("prototypes/saves")
    return scores

# create an agent
agent = DQNAgent(state_size=5,
                action_size=3,
                seed=1,
                buffer_size = args["buffer_size"],
                batch_size = args["batch_size"],
                gamma = args["gamma"],
                tau = args["tau"],
                lr = args["lr"],
                update_every = args["update_every"])

#training or just playing the game with the trained model
if args["train"]:
    env = SelfDrivingCar(render_mode=args["render"])
    scores = train_prototypes_dqn(n_episodes=args["n_episodes"],
                max_t=args["max_t"],
                eps_decay=args["eps_decay"],
                eps_start=args["eps_start"],
                eps_end=args["eps_end"],
                save_behaviour=args["save_behaviour"])
else:
    # load the weights from file
    env = SelfDrivingCar(render_mode=True)
    print("Loading the model from the file path: ", args["load_model_path"])
    agent.load_model(args["load_model_path"])
    # play the game
    state = env.reset()
    if args["save_video"]:
        video = vidmaker.Video(path=os.path.join(args["results_path"], "Proto_explainer.mp4"), fps=env.ticks, resolution=env.screen_size)
    score = 0
    n_episodes = 1000
    step = 0
    prototype_tracker = PrototypeTracker(1)
    meta_action = 2
    while True:
        
        action = agent.act(state)
        next_state,reward,done,_ = env.step(action)
        prototype_tracker.add_sample(state, action)
        meta_action = prototype_tracker.current_prototype_meta_action(10)
        env.plot_meta(meta_action)

        state = next_state
        score += reward
        step += 1
        if done:
            env.reset()
            break
        if args["save_video"]:
            video.update(pygame.surfarray.pixels3d(env.screen).swapaxes(0, 1), inverted=False) # THIS LINE
        # check if user preses button Q
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()
    if args["save_video"]:
        video.export(verbose=True)