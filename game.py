import pygame
class Cell:
    def __init__(self, state):
        self.state = state

class Grid:
    def __init__(self, x_max, y_max):
        self.grid = self._create(x_max, y_max)
    
    def _create(self, x_max, y_max):
        '''create a grid with each 2-d coordinate holds its state value as a binary number. '''
        for x, y in zip(x_max, y_max):
            state = 0
            self.grid.append({"x": x, "y": y, "state": state})
        return self.grid
    
    def update_cell(self, x, y):
        ''''''
        for cell in self.grid:
            if cell.x == x and cell.y == y:
                cell.state = 0 if cell.state == 1 else 1
            
    def state_get(self):
        '''
        returns a list of all the cell that are dangerous for the snake in binary.
        1 = means danger 
        0 = safe
        '''
        states = []
        return states
        
    
    

class SnakeGame:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        self.cells = []
        
        # init game state
        self.reset_game()
        
    def reset_game(self):
        pass
    
    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
                