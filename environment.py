import traffic_light
import math
import random
import time

class Environment():
    def __init__(self, num_traffic_lights, num_cars, max_exit_time) -> None:
        self.num_traffic_lights = num_traffic_lights
        self.num_cars = num_cars
        self.max_exit_time = max_exit_time
        self.car_simulation = CarSimulation()
        self.reset_env()
        
    def reset_env(self):
        self.roundabout = traffic_light.A14MiltonRoadRoundabout(num_traffic_lights = self.num_traffic_lights, max_exit_time = self.max_exit_time)
        self.car_simulation.reset_simulation()
        self.car_simulation.run_simulation()


    def get_state(self):
        return 

    def take_action(self, action):
        if action[1] == 0:
            self.roundabout.get_traffic_lights[action[0]].turn_off()
        else:
            self.roundabout.get_traffic_lights[action[0]].turn_on()
        for car in self.cars():
            car.update()

    def compute_reward(self):
        pass

    # def is_done(self):
    #     pass

class CarSimulation():
    def __init__(self) -> None:
        self.cars = None
        self.reset_simulation()
    
    def reset_simulation(self):
        self.cars = [traffic_light.Car() for _ in range(self.num_cars)]
        #Assign the cars to the starting lights
        for idx, car in enumerate(self.cars):
            self.cars[idx].set_current_traffic_light(random.randint(0, self.num_traffic_lights))

    def run_simulation(self):
        while True:
            for car in self.cars:
                #If traffic light is on
                #Wait 2 seconds and assign the next traffic light to the car
                if car.get_current_traffic_light().getState():
                    time.sleep(2)
                    car.set_current_traffic_light(car.get_current_traffic_light().paired_traffic_light())
                    


    


        