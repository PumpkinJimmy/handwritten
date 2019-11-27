import sys
import numpy as np
import cv2 as cv
import pygame
from pygame.locals import *
from logistic import *


SCREEN_SIZE = (448, 448)
BG_COLOR = (0, 0, 0)
PEN_COLOR = (255, 255, 255)
PEN_WIDTH = 4
CAPTION = "Handwritten Board"

with open("logistic08normalized.npz", 'rb') as f:
    chunk = np.load(f)
    w = chunk['w']

pygame.init()

screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption(CAPTION)
screen.fill(BG_COLOR)

lines = []
points = []
mouse_on = False
clock = pygame.time.Clock()
ok = True
while ok:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit()
        elif event.type == MOUSEBUTTONDOWN:
            mouse_on = True
        elif event.type == MOUSEMOTION:
            if mouse_on:
                points.append(event.pos)
        elif event.type == MOUSEBUTTONUP:
            mouse_on = False
            if len(points) >= 2:
                lines.append(points)
            points = []
        elif event.type == KEYDOWN and event.key == K_RETURN:
            if len(points) >= 2:
                lines.append(points)
            img = pygame.surfarray.array3d(screen)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).T
            img = cv.GaussianBlur(img, (3, 3), 1)
            for i in range(4):
                img = cv.pyrDown(img)
            data = normalize(img.reshape(1, 784))
            ans = logisticClassify(data, w)
            print(ans)
            lines = []
        screen.fill(BG_COLOR)
    for line in lines:
        pygame.draw.lines(screen, PEN_COLOR, False, line, PEN_WIDTH)
    if len(points) >= 2:
        pygame.draw.lines(screen, PEN_COLOR, False, points, PEN_WIDTH - 1)
    pygame.display.flip()


