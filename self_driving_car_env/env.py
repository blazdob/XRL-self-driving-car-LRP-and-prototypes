import gym
import math
import pygame
import random
import numpy as np
from pygame.locals import *

if __name__ != '__main__':
    from .pads import *
    from .sensors import Sensors, Sensor
    from .car import CarSprite
else:
    from pads import *
    from sensors import Sensors, Sensor
    from car import CarSprite
    
pads = [
    VerticalPad((10, 610)),
    HorizontalPad((30, 350)),
    SmallVerticalPad((200, 626)),
    SmallHorizontalPad((314, 500)),
    SmallVerticalPad((270, 220)),
    SmallVerticalPad((425, 380)),
    HorizontalPad((532, 105)),
    SmallHorizontalPad((540, 265)),
    SmallVerticalPad((790, 220)),
    SmallVerticalPad((650, 400)),
    SmallHorizontalPad((788, 512)),
    SmallHorizontalPad((928, 332)),
    SmallHorizontalPad((1012, 332)),
    SmallVerticalPad((925, 625)),
    VerticalPad((1123, 584)),
    VerticalPad((1123, 614)),
    HorizontalPad((680, 740)),
    HorizontalPad((440, 740)),
    HorizontalPad((240, 870)),
    SmallHorizontalPad((550, 870)),
    HorizontalPad((885, 870))
]

pads2 = [
    # outer layer
    # SmallHorizontalPad((1400, 2000)),
    VerticalPad((10, 610)),
    VerticalPad((10, 280)),
    HorizontalPad((250, 40)),
    HorizontalPad((595, 40)),
    HorizontalPad((920, 40)),
    VerticalPad((1170, 610)),
    VerticalPad((1170, 280)),
    HorizontalPad((240, 860)),
    HorizontalPad((590, 860)),
    HorizontalPad((920, 860)),

    # inner layer
    VerticalPad((190, 445)),
    SmallHorizontalPad((306, 205)),
    SmallVerticalPad((426, 318)),
    SmallHorizontalPad((540, 445)),
    SmallHorizontalPad((640, 445)),
    SmallVerticalPad((756, 333)),
    SmallHorizontalPad((869, 205)),
    SmallVerticalPad((590, 153)),
    VerticalPad((980, 445)),
    SmallHorizontalPad((869, 695)),
    SmallVerticalPad((756, 573)),
    SmallHorizontalPad((306, 685)),
    SmallVerticalPad((426, 570)),
    SmallVerticalPad((590, 743)),
]

prototypes = [
    VerticalPad((10, 610)),
    HorizontalPad((30, 350)),
    SmallVerticalPad((200, 626)),
    SmallHorizontalPad((314, 500)),
    SmallVerticalPad((270, 220)),
    SmallVerticalPad((425, 380)),
    HorizontalPad((532, 105)),
    SmallHorizontalPad((540, 265)),
    SmallVerticalPad((790, 220)),
    SmallVerticalPad((650, 400)),
    SmallHorizontalPad((788, 512)),
    SmallHorizontalPad((928, 332)),
    SmallHorizontalPad((1012, 332)),
    SmallVerticalPad((925, 625)),
    VerticalPad((1123, 584)),
    VerticalPad((1123, 614)),
    HorizontalPad((680, 740)),
    HorizontalPad((440, 740)),
    HorizontalPad((240, 870)),
    SmallHorizontalPad((550, 870)),
    HorizontalPad((885, 870))
]


class SelfDrivingCar(gym.Env):
    def __init__(self, render_mode=False, human_control=False):
        self.width = 1200
        self.height = 900
        self.screen_size = (self.width, self.height)
        self.clock = pygame.time.Clock()
        self.car = CarSprite('self_driving_car_env/images/car.png', (100, 730))
        self.pad_group = pygame.sprite.RenderPlain(*pads2)
        self.car_group = pygame.sprite.RenderPlain(self.car)
        self.sensors = Sensors(self.car.position, self.car.direction, self.pad_group)
        self.ticks = 25
        self.exit = False
        self.render_mode = render_mode
        self.human_control = human_control

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("AI Car game")
            self.font = pygame.font.Font(None, 75)

    def run(self, action):

        dt = self.clock.get_time() / 1000
        if self.human_control:
            keys = pygame.key.get_pressed()
            if keys[K_LEFT]:
                self.car.k_left = 3.5
                self.car.k_right = 0
            elif keys[K_RIGHT]:
                self.car.k_right = -3.5
                self.car.k_left = 0
            else:
                self.car.k_left = 0
                self.car.k_right = 0
        else:
            if action == 0:
                self.car.k_right = -3.5
                self.car.k_left = 0
            elif action == 1:
                self.car.k_left = 3.5
                self.car.k_right = 0
            elif action == 2:
                self.car.k_left = 0
                self.car.k_right = 0

        self.car_group.update(dt)

        collisions = pygame.sprite.groupcollide(self.car_group, self.pad_group, False, False, collided=None)
        if collisions != {}:
            self.car.speed = 0
            self.car.k_right = 0
            self.car.k_left = 0
            self.car.crashed = True

        self.sensors.update_sensors(self.car.position, self.car.direction)

        self.pad_group.update(collisions)

        if self.render_mode:
            self.render()

        self.clock.tick(self.ticks)

    def is_crashed(self):
        return self.car.crashed

    def get_sensors(self):
        return self.sensors.sens_objs

    def step(self, action):
        self.run(action)

        next_state = [-(200 / (sen.length - 10)) for sen in self.get_sensors()]
        next_state = np.array([next_state])
        
        if self.is_crashed():
            reward = -500
        else:
            reward = 10 + sum(next_state[0])
        return next_state, reward, self.is_crashed(), {}

    def render(self):
        
        # fill background
        self.screen.fill((0, 0, 0))
        # draw sensors
        self.sensors.draw(self.screen)
        
        # draw pads
        self.pad_group.draw(self.screen)
        # draw car
        self.car_group.draw(self.screen)

        # Counter Render
        pygame.display.flip()

    def reset(self):
        self.car.restart_position((100, 730))
        self.sensors = Sensors(self.car.position, self.car.direction, self.pad_group)
        return np.array([[sen.length for sen in self.get_sensors()]])

    def plot_lrp(self, attr):
        l_len = 200
        lines = []
        sensor_rel_dirs = list(map(lambda sen: sen + self.car.direction, self.sensors.sensor_dirs))
        sensor_rel_dirs = sensor_rel_dirs[::-1]
        sum_end = [self.car.position,self.car.position]
        
        for attribute, sensor_dir in zip(attr[0][0], sensor_rel_dirs):
            end_point = (self.car.position[0] + l_len * attribute * math.cos(math.radians(sensor_dir)), self.car.position[1] - l_len * attribute * math.sin(math.radians(sensor_dir)))
            line = (self.car.position, end_point)
            #sum the line with sum_end
            sum_end[1] = (sum_end[1][0] + end_point[0] - self.car.position[0], sum_end[1][1] + end_point[1] - self.car.position[1])
            # print(sum_end[1])
            lines.append(line)
            color = Color(255, 255, 255)
            pygame.draw.line(self.screen, color, line[0], line[1])
        #plot sum_end
        for i in range(len(lines)-1):
            _, end_point1 = lines[i]
            _, end_point2 = lines[i+1]
            pygame.draw.line(self.screen, Color(0,0,255), end_point1, end_point2,  width=4)
        #plot dot at which point the sum of all vector is located
        
        pygame.draw.line(self.screen, Color(255,0,0), (sum_end[0][0], sum_end[0][1]), (sum_end[1][0], sum_end[1][1]), width=7)
        pygame.display.flip()
        return 
    
    def plot_smoothed_lrp(self, attrs, smoothing):
        l_len = 150
        lines = []
        sensor_rel_dirs = list(map(lambda sen: sen + self.car.direction, self.sensors.sensor_dirs))
        sensor_rel_dirs = sensor_rel_dirs[::-1]
        sum_end = [self.car.position,self.car.position]

        # calculate mean of attrs
        attrs = np.array(attrs)
        # print("attrs: ", attrs)
        smoothed_attr = np.mean(attrs, axis=0)
        # print("smooth_attr", smoothed_attr)
        for attribute, sensor_dir in zip(smoothed_attr[0][0], sensor_rel_dirs):
            end_point = (self.car.position[0] + l_len * attribute * math.cos(math.radians(sensor_dir)), self.car.position[1] - l_len * attribute * math.sin(math.radians(sensor_dir)))
            line = (self.car.position, end_point)
            #sum the line with sum_end
            sum_end[1] = (sum_end[1][0] + end_point[0] - self.car.position[0], sum_end[1][1] + end_point[1] - self.car.position[1])
            # print(sum_end[1])
            lines.append(line)
            color = Color(255, 255, 255)
            pygame.draw.line(self.screen, color, line[0], line[1])
        #plot sum_end
        for i in range(len(lines)-1):
            _, end_point1 = lines[i]
            _, end_point2 = lines[i+1]
            pygame.draw.line(self.screen, Color(0,0,255), end_point1, end_point2,  width=4)
        #plot dot at which point the sum of all vector is located


        pygame.draw.line(self.screen, Color(255,0,0), (sum_end[0][0], sum_end[0][1]), (sum_end[1][0], sum_end[1][1]), width=7)
        pygame.display.flip()
        return

    def plot_meta(self, meta_action):
        font = pygame.font.Font('freesansbold.ttf', 32)
 
        # create a text surface object,
        # on which text is drawn on it.
        if meta_action == 2:
            text = font.render('Similar to: Driving FORWARD prototype', False, (0, 0, 255))
        elif meta_action == 1:
            text = font.render('Similar to: Turning LEFT prototype', False, (0, 0, 255))
        elif meta_action == 0:
            text = font.render('Similar to: Turning RIGHT prototype', False, (0, 0, 255))
        else:
            text = font.render('Not detected', False, (0, 0, 255))

        self.screen.blit(text, (50,50))
        pygame.display.flip()


if __name__ == '__main__':
    import sys
    game = SelfDrivingCar(render_mode=True, human_control=True)
    while True:
        # get action from keyboard input if held down
        action = 2
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 1
                if event.key == pygame.K_RIGHT:
                    action = 0

        next_state, reward, done, info = game.step(np.random.randint(0,3))
        if done:
            game.reset()