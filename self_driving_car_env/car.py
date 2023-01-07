import pygame
from math import sin, cos, pi

class CarSprite(pygame.sprite.Sprite):
    TURN_SPEED = 10

    def __init__(self, image, position):
        pygame.sprite.Sprite.__init__(self)
        self.src_image = pygame.image.load(image)
        self.position = position
        self.speed = 5
        self.direction = 0
        self.k_left = self.k_right = 0
        self.crashed = False

    def restart_position(self, position):
        self.position = position
        self.speed = 5
        self.direction = 0
        self.k_left = self.k_right = 0
        self.crashed = False


    def update(self, deltat):
        # SIMULATION
        self.direction += (self.k_right + self.k_left)
        x, y = self.position
        rad = self.direction * pi / 180
        x += -self.speed * sin(rad)
        y += -self.speed * cos(rad)
        self.position = (x, y)
        self.image = pygame.transform.rotate(self.src_image, self.direction)
        self.rect = self.image.get_rect()
        self.rect.center = self.position
