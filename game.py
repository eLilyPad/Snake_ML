import json
import os
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from dataclasses import dataclass

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

'''
reset game
reward agent
play(action) -> direction
game_iteration
is_collision
'''

# @dataclasses
# class Data:
#     # Windows
#     screen_width = 640
#     screen_height = 480

#     BLOCK_SIZE = 20
#     SPEED = 100

#     #Colours
#     WHITE = (255, 255, 255)
#     RED = (200,0,0)
#     BLUE1 = (0, 0, 255)
#     BLUE2 = (0, 100, 255)
#     BLACK = (0,0,0)



#     def __init__(self):
#         data = {}

#     def save(self, file_name = 'gamedata.json', data = {}):
#         json_string = json.dumps(data)
#         folder_path = './game'
#         file_path = os.path.join(folder_path, file_name)
#         json_file = open(file_path, 'w')

#         if not os.path.exists(folder_path):
#             os.mkdir(folder_path)

#         json_file.write(json_string)
#         json_file.close()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
class CellType(Enum):
    BG = 1
    SNAKE = 2
    FOOD = 3

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

# The size of the cells in the grid
BLOCK_SIZE = 20
# The tickrate 
SPEED = 130 

class Cell:
    def __init__(self, x, y, display, size, cell_type = CellType.BG):
        self.coord = (x, y)
        self.cell_type = cell_type
        self.size = size
        self.display = display
    
    def update_ui(self):
        if self.cell_type == CellType.BG:
            pass
        elif self.cell_type == CellType.SNAKE:
            self._draw_snake()
        elif self.cell_type == CellType.FOOD:
            self._draw_food()

    def _draw_snake(self):
        self._draw_cell(BLUE1, self.size)
        self._draw_cell(BLUE2, self.size, self.size - 8)

    def _draw_food(self):
        self._draw_cell(RED, self.size)

    def _draw_cell(self, colour, scale):
        pygame.draw.rect(self.display, colour, pygame.Rect(self.coord.x, self.coord.y, scale, scale))


class CellGrid:
    def __init__(self, display, scale = 1, width = 64, height = 48):
        self.scale = scale
        self.width = width
        self.height = height
        self.display = display

    def create_grid(self):
        cells = []
        for x, y in zip(self.width, self.height):
            cells.append(Cell(x, y, self.display))
        return cells

    def get_cell(self, x, y):
        pass

    def set_cell_state(self, state):
        pass


class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        # init game state
        self.reset_game()
        
    def reset_game(self):
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head, 
            Point(self.head.x-BLOCK_SIZE, self.head.y),
            Point(self.head.x-(2*BLOCK_SIZE), self.head.y)
        ]
        
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self.last_food_time = 0

        self._place_food()
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.last_food_time = 0
            self._place_food()
        
    def play_step(self, action):
        self.frame_iteration += 1 
        self.last_food_time += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        good = 10 - (self.last_food_time / 100)
        bad = -(10 + (self.last_food_time / 100))
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = bad
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = good
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt = None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def point_from_head(self, direction, distance = 1):
        head = self.snake[0]
        offset = BLOCK_SIZE * distance
        
        if direction == Direction.LEFT:
            return Point(head.x - offset, head.y)
        elif direction == Direction.RIGHT:
            return Point(head.x + offset, head.y)
        elif direction == Direction.UP:
            return Point(head.x, head.y - offset)
        elif direction == Direction.DOWN:
            return Point(head.x, head.y + offset)
          
    def _move(self, action):
        # [straight, left, right]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[index] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_index = (index + 1) % 4
            new_direction = clock_wise[next_index] # right turn
        else:
            # [0, 0, 1]
            next_index = (index - 1) % 4
            new_direction = clock_wise[next_index] # left turn
        
        self.direction = new_direction
            
    

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
