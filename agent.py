import torch
import network
import CacheRecall
import numpy as np
import random
from itertools import count
import matplotlib.pyplot as plt
import torch.optim as optim
import time


class Agent():
    def __init__(self, BATCH_SIZE, MEMORY_SIZE, GAMMA, input_dim, output_dim_traffic_light, output_dim_on_off, action_dim_traffic_light, action_dim_on_off, EPS_START, EPS_END, EPS_DECAY_VALUE, lr, TAU) -> None:
        self.BATCH_SIZE=BATCH_SIZE
        self.GAMMA = GAMMA
        self.MEMORY_SIZE=MEMORY_SIZE
        self.action_dim_traffic_light = action_dim_traffic_light
        self.action_dim_on_off = action_dim_on_off
        self.EPS_START=EPS_START
        self.EPS_END=EPS_END
        self.EPS_DECAY_VALUE=EPS_DECAY_VALUE
        self.eps = EPS_START
        self.TAU = TAU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.episode_durations = []
        #Create the cache recall memory
        self.cache_recall = CacheRecall.CacheRecall(memory_size=MEMORY_SIZE)
        self.policy_net = network.DQN(input_dim=input_dim, output_dim_traffic_light=output_dim_traffic_light, output_dim_on_off=output_dim_on_off).to(self.device)
        self.target_net = network.DQN(input_dim=input_dim, output_dim_traffic_light=output_dim_traffic_light, output_dim_on_off=output_dim_on_off).to(self.device)
        #No need to calculate gradients for target net parameters as these are periodically copied from the policy net
        for param in self.target_net.parameters():
            param.requires_grad = False
        #Copy the initial parameters from the policy net to the target net to align them at the start
        #Diff
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.steps_done = 0
    
    @torch.no_grad
    def take_action(self, state):
        #Decay and cap the epsilon value
        self.eps = self.eps*self.EPS_DECAY_VALUE
        self.eps = max(self.eps, self.EPS_END)
        #Take a random action
        if self.eps < np.random.rand():
            state = state[None, :]
            action_1, action_2 = self.policy_net(state)
            action_idx_1 = torch.argmax(action_1, dim=1).item()
            action_idx_2 = torch.argmax(action_2, dim=1).item()
        #Else take a greedy action
        else:
            action_idx_1 = random.randint(0, self.action_dim_traffic_light-1)
            action_idx_2 = random.randint(0, self.action_dim_on_off-1)
        self.steps_done += 1
        return action_idx_1, action_idx_2
    
    def plot_durations(self):
        plt.figure(1)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        #Plot the durations
        plt.plot(durations_t.numpy())
        # Take 100 episode averages of the durations and plot them too, to show a running average on the graph
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.savefig(self.network_type+'_training.png')


    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        #Update the parameters in the target network
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
            self.target_net.load_state_dict(target_net_state_dict)


    def optimize_model(self):
        if len(self.cache_recall) < self.BATCH_SIZE:
            return
        batch = self.cache_recall.recall(self.BATCH_SIZE)
        batch = [*zip(*batch)]
        state = torch.stack(batch[0])
        #batch[1] gives us the next_state after the action, we want to create a mask to filter out the states where we end the run (i.e. the flappy bird dies). the end of the run will give a state of None which cannot be inputted into the network
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[1])), device=self.device, dtype=torch.bool)
        #Grab the next states that are not final states
        non_final_next_states = torch.stack([s for s in batch[1] if s is not None])
        #Grab the action and the reward
        action_1 = torch.stack(batch[2])
        action_2 = torch.stack(batch[3])
        reward = torch.cat(batch[4])
        #TODO: REVIEW THIS
        next_state_action_values_action_1 = torch.zeros(self.BATCH_SIZE, dtype=torch.float32, device=self.device)
        next_state_action_values_action_2 = torch.zeros(self.BATCH_SIZE, dtype=torch.float32, device=self.device)
        #Get the Q values from the policy network and then get the Q value for the given action taken
        #[32,1]
        state_output_1, state_output_2 = self.policy_net(state)
        state_action_values_action_1 = state_output_1.gather(1, action_1)
        state_action_values_action_2 = state_output_2.gather(1, action_2)
        # print(self.target_net(non_final_next_states).max(1, keepdim=True)[0].size())
        #Use the target network to get the maximum Q for the next state across all the actions
        with torch.no_grad():
            next_state_output_1, next_state_output_2 = self.target_net(non_final_next_states)
            next_state_action_values_action_1[non_final_mask] = next_state_output_1.max(1)[0]
            next_state_action_values_action_2[non_final_mask] = next_state_output_2.max(1)[0]
        #Calcuate the expected state action values as the per equation we use
        expected_state_action_values_action_1 = (next_state_action_values_action_1 * self.GAMMA) + reward
        expected_state_action_values_action_2 = (next_state_action_values_action_2 * self.GAMMA) + reward
        #Using the L1 Loss, calculate the difference between the expected and predicted values and optimize the policy network only
        loss_fn_1 = torch.nn.SmoothL1Loss()
        loss_fn_2 = torch.nn.SmoothL1Loss()
        # print(state_action_values.size())
        #print(expected_state_action_values.size())
        loss1 = loss_fn_1(state_action_values_action_1, expected_state_action_values_action_1.unsqueeze(1))
        loss2 = loss_fn_2(state_action_values_action_2, expected_state_action_values_action_2.unsqueeze(1))
        loss = loss1 + loss2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def train(self, episodes, env):
        self.steps_done = 0
        for episode in range (episodes):
            env.reset_env()
            state = env.get_state()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            for c in count():
                action_1, action_2 = self.take_action(state)
                reward = env.take_action(action_1, action_2)
                time.sleep(5)
                reward = torch.tensor([reward], device=self.device)
                print(reward)
                action_1 = torch.tensor([action_1], device=self.device)
                action_2 = torch.tensor([action_2], device=self.device)
                next_state = env.get_state()
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                done = False
                if done:
                    next_state = None
                #Cache a tuple of the date
                self.cache_recall.cache((state, next_state, action_1, action_2, reward, done))
                #Set the state to the next state
                state = next_state
                #Optimize the model and update the target network 
                self.optimize_model()
                self.update_target_network()
                if done:
                    #Update the number of durations for the episode
                    self.episode_durations.append(c+1)
                    #Plot them and save the networks
                    self.plot_durations()
                    print("EPS: {}".format(self.eps))
                    print("Durations: {}".format(c+1))
                    print("Score: {}".format(env.score()))
                    torch.save(self.target_net.state_dict(), self.network_type+'_target_net.pt')
                    torch.save(self.policy_net.state_dict(), self.network_type+'_policy_net.pt')
                    #Start a new episode
                    break


