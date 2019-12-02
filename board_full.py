import sys
import numpy as np
import cv2 as cv
import pygame
import matplotlib.pyplot as plt
from pygame.locals import *
from softmax import normalize, h


SCREEN_SIZE = (448, 448)
BG_COLOR = (0, 0, 0)
PEN_COLOR = (255, 255, 255)
PEN_WIDTH = 20
CAPTION = "Handwritten Board"

with open("result/softmaxNormalized.npz", 'rb') as f:
    chunk = np.load(f)
    w = chunk['w']

pygame.init()

screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption(CAPTION)
screen.fill(BG_COLOR)

lines = []
points = []
anti = []
mouse_on = False
clock = pygame.time.Clock()
ok = True
while ok:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == QUIT:
            with open("anti.npz", 'wb') as f:
                np.savez(f, anti=anti)
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
        elif event.type == KEYDOWN and \
                (event.key == K_RETURN or event.key == K_s):
            if len(points) >= 2:
                lines.append(points)
            img = pygame.surfarray.array3d(screen)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).T
            img = cv.GaussianBlur(img, (3, 3), 1)
            cv.imshow("img", img)
            cv.waitKey(0)
            for i in range(4):
                img = cv.pyrDown(img)
            if (event.key == K_s):
                anti.append(img.copy())
            data = np.c_[normalize(img.reshape(1, 784)), 1]
            ans = np.argmax(h(data, w))
            print(ans)
            lines = []
        screen.fill(BG_COLOR)
    for line in lines:
        pygame.draw.lines(screen, PEN_COLOR, False, line, PEN_WIDTH)
    if len(points) >= 2:
        pygame.draw.lines(screen, PEN_COLOR, False, points, PEN_WIDTH - 1)
    pygame.display.flip()
