import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate. must be smaller then ones
        self.memory = deque(maxlen = MAX_MEMORY)
        self.model = Linear_QNet(11, 23, 3)
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

        if os.path.exists('model/model.pth'):
            self.model.load_state_dict(torch.load('model/model.pth'))
            self.model.eval()

    def get_state(self, game):
        block_size = 20
        head = game.snake[0]
        point_left = Point(head.x - block_size, head.y)
        point_right = Point(head.x + block_size, head.y)
        point_up = Point(head.x, head.y - block_size)
        point_down = Point(head.x, head.y + block_size)

        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (direction_right and game.is_collision(point_right)) or
            (direction_left and game.is_collision(point_left)) or
            (direction_up and game.is_collision(point_up)) or
            (direction_down and game.is_collision(point_down)),
            
            # Danger Right
            (direction_up and game.is_collision(point_right)) or
            (direction_down and game.is_collision(point_left)) or
            (direction_left and game.is_collision(point_up)) or
            (direction_right and game.is_collision(point_down)),
            
            # Danger Left
            (direction_down and game.is_collision(point_right)) or
            (direction_up and game.is_collision(point_left)) or
            (direction_left and game.is_collision(point_up)) or
            (direction_right and game.is_collision(point_down)),

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
        self.epsilon = 0#80 - self.n_games
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
                agent.model.save()

            print('Game', agent.n_games, 'Score:', score, 'Record:', best_score, 'Reward:', reward)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()