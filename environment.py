import traffic_light
import math
import random
import time
import threading

class Environment():
    def __init__(self, num_traffic_lights, max_num_cars, max_exit_time) -> None:
        self.num_traffic_lights = num_traffic_lights
        self.max_num_cars = max_num_cars
        self.max_exit_time = max_exit_time
        self.car_simulation = None
        #self.reset_env()
        self.threads = []
        
    def reset_env(self):
        for thread in self.threads:
            thread.join()
        self.threads = []
        num_cars = random.randint(1, self.max_num_cars)
        self.car_simulation = CarSimulation(num_cars, self.num_traffic_lights)
        self.roundabout = traffic_light.A14MiltonRoadRoundabout(num_traffic_lights = self.num_traffic_lights, max_exit_time = self.max_exit_time)
        self.car_simulation.reset_simulation(self.roundabout.traffic_lights)
        #Start running the simulation in a seperate thread
        thread = threading.Thread(target=self.car_simulation.watch_simulation, daemon=True)
        thread.start()
        self.threads.append(thread)
        thread = threading.Thread(target=self.car_simulation.run_simulation, daemon=True)
        thread.start()
        self.threads.append(thread)
        

    def is_done(self):
        for car in self.car_simulation.getCars():
            if car.get_is_exited() == False:
                return False
        return True

    def get_state(self):
        traffic_light_array = []
        for traffic_light in self.roundabout.traffic_lights:
            if traffic_light.getState():
                traffic_light_array.append(1)
            else:
                traffic_light_array.append(0)
        num_cars_waiting_at_start = 0
        num_cars_waiting_in_roundabout = 0
        num_cars_exited = 0
        for car in self.car_simulation.getCars():
            if car.get_is_exited():
                num_cars_exited += 1
            elif car.get_current_traffic_light().is_in_roundabout:
                num_cars_waiting_in_roundabout += 1
            else:
                num_cars_waiting_at_start += 1
        return traffic_light_array + [num_cars_waiting_at_start, num_cars_waiting_in_roundabout, num_cars_exited]

    def take_action(self, action_1, action_2):
        if action_2 == 0:
            self.roundabout.get_traffic_lights()[action_1].turn_off()
        else:
            self.roundabout.get_traffic_lights()[action_1].turn_on()

    def compute_reward(self):
        #We define the reward to be +1 fir nay cars waiting at the start traffic lights -1 for any cars waiting inside the roundabout I.E. at the inside lights
        total_reward = 0
        for car in self.car_simulation.getCars():
            if car.get_is_exited() and not car.has_stopped_in_roundabout:
                total_reward += 10
            elif car.get_is_exited():
                total_reward += 5
            elif car.get_current_traffic_light().is_in_roundabout:
                total_reward += -1
            else:
                total_reward += 0

        return total_reward


class CarSimulation():
    def __init__(self, num_cars, num_traffic_lights) -> None:
        self.num_cars = num_cars
        self.num_exits = num_traffic_lights
        self.num_traffic_lights = num_traffic_lights*2
        self.cars = None
        self.traffic_lights = None
        # self.reset_simulation()
    
    def reset_simulation(self, traffic_lights):
        print('RESET')
        self.cars = [traffic_light.Car() for _ in range(self.num_cars)]
        self.traffic_lights = traffic_lights
        #Assign the cars to the starting lights and assign the exit
        for idx, car in enumerate(self.cars):
            self.cars[idx].set_current_traffic_light(traffic_lights[random.randint(0, (self.num_traffic_lights/2)-1)])
            self.cars[idx].set_exit_to_take(self.num_exits, self.traffic_lights.index(self.cars[idx].get_current_traffic_light()))
    
    def all_cars_exit(self):
        for idx, car in enumerate(self.cars):
            if car.get_is_exited() == False:
                return False
        return True

    #Debugging function to run in a seperate thread
    def watch_simulation(self):
        while True:
            time.sleep(7)
            # car = self.cars[0]
            for idx, traffic_light in enumerate(self.traffic_lights):
                traffic_light_status = 'on' if traffic_light.getState() else 'off'
                traffic_light_roundabout = traffic_light.is_in_roundabout
                print('Traffic Light {} is {} and is in roundabout? {}'.format(idx, traffic_light_status, traffic_light_roundabout))        
            for idx, car in enumerate(self.cars):
                traffic_light = self.traffic_lights.index(car.get_current_traffic_light())
                traffic_light_status = car.get_current_traffic_light().getState()
                traffic_light_status = 'on' if traffic_light_status else 'off'
                print('Car {} is currently at traffic light {}, has exit {}, which is {} and has exited? {}'.format(idx, traffic_light, car.get_exit_to_take(), traffic_light_status, car.get_is_exited()))
            if self.all_cars_exit():
                return True
            


    def getCars(self):
        return self.cars

    def run_simulation(self):
        while True:
            for car in self.cars:
                # print('hi')
                #If traffic light is on
                #Wait 2 seconds and assign the next traffic light to the car
                # print(type(car.get_current_traffic_light()))
                while car.get_current_traffic_light().getState() and not car.get_is_exited():
                    #Exit the car if it is at the correct roundabout light and that light is on
                    if car.get_current_traffic_light().is_in_roundabout:
                        print(self.traffic_lights.index(car.get_current_traffic_light().paired_traffic_light))
                        idx_current_car = self.traffic_lights.index(car.get_current_traffic_light().paired_traffic_light)
                        # if idx_current_car == len(self.traffic_lights) -1:
                        #     idx_current_car = 0
                        # else:
                        #     idx_current_car = math.floor(idx_current_car)
                        print(car.get_current_traffic_light().paired_traffic_light.is_in_roundabout)
                        print(idx_current_car)
                        print(car.get_exit_to_take())
                        if car.get_exit_to_take() == idx_current_car and car.get_current_traffic_light().getState():
                            car.set_is_exited(True)
                            continue
                    time.sleep(0.5)
                    print('Setting traffic light')
                    car.set_current_traffic_light(car.get_current_traffic_light().paired_traffic_light)
                    if not car.get_current_traffic_light().getState() and car.get_current_traffic_light().is_in_roundabout:
                        car.has_stopped_in_roundabout = True
            if self.all_cars_exit():
                return True

# env = Environment(5, 6, 3)
# env.reset_env()