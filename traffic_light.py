import enum
import random


class TrafficLight():
    def __init__(self, paired_traffic_light, blocking_traffic_light) -> None:
        self.isOn = False
        self.paired_traffic_light = paired_traffic_light
        self.blocking_traffic_light = blocking_traffic_light
        self.is_in_roundabout = False

    def setPairedTrafficLight(self, paired_traffic_light):
        self.paired_traffic_light = paired_traffic_light

    def setBlockingTrafficLight(self, blocking_traffic_light):
        self.blocking_traffic_light = blocking_traffic_light

    def getState(self):
        return self.isOn
    
    def turn_on(self):
        self.isOn = True
        self.blocking_traffic_light.turn_off()
    
    def turn_off(self):
        self.isOn = False

# trafficlight = TrafficLight()
# print(trafficlight.state)

class Car():
    def __init__(self) -> None:
        self.timer = 0
        self.is_waiting = True
        self.is_stuck = False
        self.curr_traffic_light = None
        self.exit = None
        self.exited = False
        self.has_stopped_in_roundabout = False

    def set_exit_to_take(self, num_exits, starting_traffic_light_num):
        exit_to_take = random.randint(0, num_exits-1)
        while exit_to_take == starting_traffic_light_num:
            exit_to_take = random.randint(0, num_exits-1)
        # #Hacky hack for test
        # exit_to_take = 0
        self.exit = exit_to_take

    def get_is_exited(self):
        return self.exited
    
    def set_is_exited(self, exited):
        self.exited = exited

    def get_exit_to_take(self):
        return self.exit
        
    def get_is_waiting(self):
        return self.is_waiting

    def get_current_traffic_light(self):
        return self.curr_traffic_light

    def set_current_traffic_light(self, curr_traffic_light):
        self.curr_traffic_light = curr_traffic_light
    
    def get_is_stuck(self):
        return self.is_stuck

    # def update(self):
    #     if self.curr_traffic_light.getState():
            

        
class A14MiltonRoadRoundabout():
    def __init__(self, num_traffic_lights, max_exit_time) -> None:
        self.max_exit_time = max_exit_time
        self.traffic_lights_enter = [TrafficLight(None, None) for i in range(num_traffic_lights)]
        self.traffic_lights_roundabout = [TrafficLight(None, None) for i in range(num_traffic_lights)]
        for trafficlight in self.traffic_lights_roundabout:
            trafficlight.is_in_roundabout = True
        self.traffic_lights = self.traffic_lights_enter + self.traffic_lights_roundabout
        traffic_lights_enter_modified = self.traffic_lights_enter[1:]
        traffic_lights_enter_modified.append(self.traffic_lights_enter[0])
        traffic_lights_roundabout_modified = [self.traffic_lights_roundabout[-1]] + self.traffic_lights_roundabout[:-1]
        # print(traffic_lights_roundabout_modified)
        #Setup Paired traffic lights
        print(self.traffic_lights)
        for i in range(num_traffic_lights):
            self.traffic_lights_enter[i].setPairedTrafficLight(self.traffic_lights_roundabout[i])
            self.traffic_lights_enter[i].setBlockingTrafficLight(traffic_lights_roundabout_modified[i])
            print(self.traffic_lights_enter[i].paired_traffic_light)
            self.traffic_lights_roundabout[i].setPairedTrafficLight(traffic_lights_enter_modified[i])
            self.traffic_lights_roundabout[i].setBlockingTrafficLight(traffic_lights_enter_modified[i])
        # self.traffic_light_dict = {}
        # for i in range(num_traffic_lights*2):
        #     self.traffic_light_dict.update({i: self.traffic_lights[i]})
    
    def get_traffic_lights(self):
        return self.traffic_lights




