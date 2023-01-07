import numpy as np
import random 
from collections import namedtuple, deque 

##Importing the model (function approximator for Q-table)
from models import QNetwork, QNetworkLRP

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("mps")

class DQNAgent():
    """Interacts with and learns from environment."""
    
    def __init__(self, state_size, action_size, seed, buffer_size = int(1e5), batch_size = 64, gamma = 0.99, tau = 1e-3, lr = 5e-4, update_every = 4, explanation_type=None):
        """Initialize an Agent object.
        
        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        
        #Q- Network
        if explanation_type == "LRP":
            self.qnetwork_local = QNetworkLRP(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkLRP(state_size, action_size, seed).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=self.lr)
        
        # Replay memory 
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self, state, action, reward, next_step, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step+1)% self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory)>self.batch_size:
                experience = self.memory.sample()
                self.learn(experience)

    def act(self, state, eps = 0):
        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        #Epsilon -greedy action selction

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        ## TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        # Local model is one which we need to train so it's in training mode
        self.qnetwork_local.train()
        # Target model is one with which we need to get our target so it's in evaluation mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function
        self.qnetwork_target.eval()
        #shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states).gather(1,actions)
    
        with torch.no_grad():
            labels_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (self.gamma * labels_next*(1-dones))
        
        loss = criterion(predicted_targets,labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local,self.qnetwork_target,)
            
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1-self.tau)*target_param.data)

    def load_model(self, path):
        self.qnetwork_local.load_state_dict(torch.load(path))
        self.qnetwork_target.load_state_dict(torch.load(path))
    
    def explain_step(self, rule, state, action, reward, next_state, done, projecing=True):
        """Save experience in replay memory, and use random sample from buffer to learn.

        Returns the explanation for the given state as per current policy

        Params
        =======
            state (array_like): current state
        """

        self.qnetwork_local.eval()
        transformed_state = torch.tensor(state.copy()).float().unsqueeze(0).to(device)
        transformed_state.requires_grad_(True)

        def compute_explanation(rule, ax_, title=None, postprocess=None, pattern=None, cmap='seismic', projecing=True): 

            # # # # For the interested reader:
            # This is where the LRP magic happens.
            # Reset gradient
            transformed_state.grad = None

            # Forward pass with rule argument to "prepare" the explanation
            y_hat = self.qnetwork_local.forward(transformed_state, explain=True, rule=rule, pattern=pattern)
            # Choose argmax
            # print(y_hat)
            # print("hat_before: ", y_hat)
            # y_hat = y_hat[torch.arange(transformed_state.shape[0]), y_hat.max(1)[1]]
            # print(y_hat)
            y_hat = y_hat.sum()
            # Backward pass (compute explanation)
            y_hat.backward()
            attr = transformed_state.grad

            if postprocess:  # Used to compute input * gradient
                with torch.no_grad():
                    attr = postprocess(attr.cpu())
            # get the index of the biggest absolute number:
            if projecing and rule != "alpha1beta0":
                attr = project(attr.cpu().numpy())
            else:
                attr = attr.cpu().numpy()
            # print(rule, attr)
            return attr

        # print("transformed_state: ", transformed_state)
        if rule == "input*gradient":
            attr = compute_explanation("gradient", None, title="input*gradient", postprocess = lambda attribution: attribution * state)
        else:
            attr = compute_explanation(rule, None, title=rule)
        return attr
        
class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])
        self.seed = random.seed(seed)
        
    def add(self,state, action, reward, next_state,done):
        """Add a new experience to memory."""
        e = self.experiences(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory,k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)