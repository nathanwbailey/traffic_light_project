import agent
import environment

NUM_TRAFFIC_LIGHTS=5
NUM_CARS = 10

env = environment.Environment(num_traffic_lights=NUM_TRAFFIC_LIGHTS, num_cars=NUM_CARS, max_exit_time=5)

state = env.get_state()
print(state)
len_state = len(state)
print(len_state)

#Create the agent and train it
agent = agent.Agent(BATCH_SIZE=32, MEMORY_SIZE=100000, GAMMA=0.99, input_dim=len_state, output_dim_traffic_light=NUM_TRAFFIC_LIGHTS, output_dim_on_off=2, action_dim_traffic_light=NUM_TRAFFIC_LIGHTS, action_dim_on_off=2, EPS_START=1.0, EPS_END=0.05, EPS_DECAY_VALUE=0.999995, TAU = 0.005, lr = 1e-4)

agent.train(episodes=10000000, env=env)
