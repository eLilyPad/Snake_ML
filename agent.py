import torch
import random
import numpy as np
from enum import Enum
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Danger(Enum):
    STRAIGHT = 1
    LEFT = 2
    RIGHT = 3

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate. must be smaller then ones
        self.memory = deque(maxlen = MAX_MEMORY)

        self.q_net_in = 11 # number of input states from the neural network
        self.model = Linear_QNet(self.q_net_in, 23, 3)
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

        self.model_name = f'model/model_{self.q_net_in}.pth'
        if os.path.exists(self.model_name):
            self.model.load_state_dict(torch.load(self.model_name))
            self.model.eval()

    def get_state(self, game):
        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        def danger_check(self, direction, distance = 1):
            point_left = game.point_from_head(Direction.LEFT, distance)
            point_right = game.point_from_head(Direction.RIGHT, distance)
            point_up = game.point_from_head(Direction.UP, distance)
            point_down = game.point_from_head(Direction.DOWN, distance)

            col_points = [direction_right, direction_left, direction_up, direction_down] # Danger Straight
            if direction == Danger.LEFT:
                col_points = [direction_down, direction_up, direction_left, direction_right]
            elif direction == Danger.RIGHT:
                col_points = [direction_up, direction_down, direction_left, direction_right]

            return (
                (col_points[0] and game.is_collision(point_right)) or
                (col_points[1] and game.is_collision(point_left)) or
                (col_points[2] and game.is_collision(point_up)) or
                (col_points[3] and game.is_collision(point_down))
            )

        state = [
            # Danger Straight
            danger_check(Danger.STRAIGHT, 1),
            # danger_check(Danger.STRAIGHT, 2),
            
            # Danger Right
            danger_check(Danger.RIGHT, 1),
            # danger_check(Danger.RIGHT, 2),
            
            # Danger Left
            danger_check(Danger.LEFT, 1),
            # danger_check(Danger.LEFT, 2),

            # Move Directions
            direction_left,
            direction_right,
            direction_up,
            direction_down,

            # Food Location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right 
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down
            # knows the foods direction but only knows if danger is directly next to it 
        ]
        
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
        
        # same as 
        # for state, action, reward, next_state, game_over in mini_sample
        #   self.trainer.train_step(state, action, reward, next_state, game_over)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # start off with random moves for exploration
        self.epsilon = 0#20 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        prev_state = agent.get_state(game)

        final_move = agent.get_action(prev_state)

        reward, game_over, score = game.play_step(final_move)
        next_state = agent.get_state(game)

        agent.train_short_memory(prev_state, final_move, reward, next_state, game_over)

        agent.remember(prev_state, final_move, reward, next_state, game_over)

        if game_over:
            # train_long_memory, plot result
            game.reset_game()
            agent.n_games += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                agent.model.save(agent.model_name)

            print('Game', agent.n_games, 'Score:', score, 'Record:', best_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()