import enum
import random


class TrafficLight():
    def __init__(self, paired_traffic_light) -> None:
        self.isOn = False
        self.paired_traffic_light = paired_traffic_light
        self.is_in_roundabout = False

    def setPairedTrafficLight(self, paired_traffic_light):
        self.paired_traffic_light = paired_traffic_light

    def getState(self):
        return self.isOn
    
    def turn_on(self):
        self.isOn = True
        self.paired_traffic_light.turn_off()
    
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

    def set_exit_to_take(self, num_exits):
        self.exit = random.randint(0, num_exits)

    def get_exit_to_take(self, num_exits):
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
        self.traffic_lights_enter = [TrafficLight(None) for i in range(num_traffic_lights)]
        self.traffic_lights_roundabout = [TrafficLight(None) for i in range(num_traffic_lights)]
        for trafficlight in self.traffic_lights_roundabout:
            trafficlight.is_in_roundabout = True
        self.traffic_lights = self.traffic_lights_enter + self.traffic_lights_roundabout
        traffic_lights_enter_modified = self.traffic_lights_enter[1:]
        traffic_lights_enter_modified.append(self.traffic_lights_enter[0])
        #Setup Paired traffifc lights
        for i in range(num_traffic_lights):
            self.traffic_lights_enter[i].setPairedTrafficLight(self.traffic_lights_roundabout[i])
            self.traffic_lights_roundabout[i].setPairedTrafficLight(traffic_lights_enter_modified[i])
        # self.traffic_light_dict = {}
        # for i in range(num_traffic_lights*2):
        #     self.traffic_light_dict.update({i: self.traffic_lights[i]})
    
    def get_traffic_lights(self):
        return self.traffic_lights




