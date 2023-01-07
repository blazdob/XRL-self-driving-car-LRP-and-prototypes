import pygame
import math
from pygame.locals import *
from shapely.geometry import LineString

class Sensors:
    def __init__(self, car_pos, car_dir, pad_group):
        self.pad_group = pad_group
        self.sensor_dirs = [30, 60, 90, 120, 150]
        self.sens_objs = []
        self.update_sensors(car_pos, car_dir)

    def update_sensors(self, car_pos, car_dir):
        self.sens_objs = []
        l_len = 10000
        sensor_rel_dirs = list(map(lambda sen: sen + car_dir, self.sensor_dirs))
        for s in sensor_rel_dirs:
            inf_line = (car_pos, (car_pos[0] + l_len * math.cos(math.radians(s)), car_pos[1] - l_len * math.sin(math.radians(s))))
            self.sens_objs.append(Sensor(car_pos, Sensors.get_closest_pad_intersection(car_pos, inf_line, self.pad_group)))

    def draw(self, canvas):
        for sensor in self.sens_objs:
            sensor.draw(canvas)

    @staticmethod
    def get_closest_pad_intersection(car_pos, line, pad_group):
        res = []
        x, y = car_pos
        for pad in pad_group:
            res.append(line_intersection(line, (pad.rect.topleft, pad.rect.bottomleft)))
            res.append(line_intersection(line, (pad.rect.topleft, pad.rect.topright)))
            res.append(line_intersection(line, (pad.rect.topright, pad.rect.bottomright)))
            res.append(line_intersection(line, (pad.rect.bottomleft, pad.rect.bottomright)))
        return min(res, key=lambda point: distance(x, y, point[0], point[1]))


class Sensor:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.length = distance(start[0], start[1], end[0], end[1])
        if self.length <= 40:
            self.color = Color(255, 0, 0)
        elif 20 < self.length <= 150:
            self.color = Color(255, 69, 0)
        else:
            self.color = Color(50, 205, 50)

    def draw(self, canvas):
        pygame.draw.line(canvas, self.color, self.start, self.end)


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def line_intersection(line1, line2):
    line1 = LineString(line1)
    line2 = LineString(line2)

    intersection = line1.intersection(line2)
    if intersection.is_empty:
        return 100000, 100000
    else:
        return intersection.x, intersection.y