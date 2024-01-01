import traffic_light
import math
import random
import time
import threading

class Environment():
    def __init__(self, num_traffic_lights, num_cars, max_exit_time) -> None:
        self.num_traffic_lights = num_traffic_lights
        self.num_cars = num_cars
        self.max_exit_time = max_exit_time
        self.car_simulation = CarSimulation(num_cars, num_traffic_lights)
        self.reset_env()
        
    def reset_env(self):
        self.roundabout = traffic_light.A14MiltonRoadRoundabout(num_traffic_lights = self.num_traffic_lights, max_exit_time = self.max_exit_time)
        self.car_simulation.reset_simulation(self.roundabout.traffic_lights)
        #Start running the simulation in a seperate thread
        thread = threading.Thread(target=self.car_simulation.run_simulation, daemon=True)
        thread.start()

    def get_state(self):
        num_cars_waiting_at_start = 0
        num_cars_waiting_in_roundabout = 0
        for car in self.car_simulation.getCars():
            if car.get_current_traffic_light().is_in_roundabout:
                num_cars_waiting_in_roundabout += 1
            else:
                num_cars_waiting_at_start += 1
        return [num_cars_waiting_at_start, num_cars_waiting_in_roundabout]

    def take_action(self, action_1, action_2):
        if action_2 == 0:
            self.roundabout.get_traffic_lights()[action_1].turn_off()
        else:
            self.roundabout.get_traffic_lights()[action_1].turn_on()
        return self.compute_reward()

    def compute_reward(self):
        #We define the reward to be +1 fir nay cars waiting at the start traffic lights -1 for any cars waiting inside the roundabout I.E. at the inside lights
        total_reward = 0
        for car in self.car_simulation.getCars():
            if car.get_current_traffic_light().is_in_roundabout:
                total_reward += -1
            else:
                total_reward += 1
        return total_reward


class CarSimulation():
    def __init__(self, num_cars, num_traffic_lights) -> None:
        self.num_cars = num_cars
        self.num_traffic_lights = num_traffic_lights
        self.cars = None
        # self.reset_simulation()
    
    def reset_simulation(self, traffic_lights):
        self.cars = [traffic_light.Car() for _ in range(self.num_cars)]
        #Assign the cars to the starting lights
        for idx, car in enumerate(self.cars):
            self.cars[idx].set_current_traffic_light(traffic_lights[random.randint(0, self.num_traffic_lights)])

    #Debugging function to run in a seperate thread
    def watch_simulation():
        pass

    
    def getCars(self):
        return self.cars

    def run_simulation(self):
        while True:
            for car in self.cars:
                # print('hi')
                #If traffic light is on
                #Wait 2 seconds and assign the next traffic light to the car
                # print(type(car.get_current_traffic_light()))
                if car.get_current_traffic_light().getState():
                    time.sleep(0.5)
                    car.set_current_traffic_light(car.get_current_traffic_light().paired_traffic_light)
                    

# env = Environment(5, 6, 3)
# env.reset_env()

    


        