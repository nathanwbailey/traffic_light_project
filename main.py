import agent
import environment

NUM_TRAFFIC_LIGHTS=4
MAX_NUM_CARS = 1

env = environment.Environment(num_traffic_lights=NUM_TRAFFIC_LIGHTS, max_num_cars=MAX_NUM_CARS, max_exit_time=5)

# state = env.get_state()
len_state = 3+NUM_TRAFFIC_LIGHTS*2

#Create the agent and train it
agent = agent.Agent(BATCH_SIZE=32, MEMORY_SIZE=100000, GAMMA=0.99, input_dim=len_state, output_dim_traffic_light=NUM_TRAFFIC_LIGHTS*2, output_dim_on_off=2, action_dim_traffic_light=NUM_TRAFFIC_LIGHTS*2, action_dim_on_off=2, EPS_START=1.0, EPS_END=0.05, EPS_DECAY_VALUE=0.99995, TAU = 0.005, lr = 1e-4)

agent.train(episodes=10000000, env=env)
