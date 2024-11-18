import numpy as np
import random
from itertools import product
from copy import deepcopy
from math import sqrt, log

class GameState:
    def __init__(self, board, selected_position, selected_piece, available_pieces):
        self.board = board
        self.selected_position = selected_position
        self.selected_piece = selected_piece
        self.available_pieces = available_pieces

    def check_wins(self):
        # Check rows, columns, and diagonals
        for col in range(BOARD_COLS):
            if self._check_line([self.board[row][col] for row in range(BOARD_ROWS)]):
                return True
        
        for row in range(BOARD_ROWS):
            if self._check_line([self.board[row][col] for col in range(BOARD_COLS)]):
                return True
            
        if self._check_line([self.board[i][i] for i in range(BOARD_ROWS)]) or self._check_line([self.board[i][BOARD_ROWS - i - 1] for i in range(BOARD_ROWS)]):
            return True

        # Check 2x2 sub-grids
        if self._check_2x2_subgrid_win():
            return True
        
        return False
    
    def _check_line(self, line):
        if 0 in line:
            return False  # Incomplete line
        characteristics = np.array([pieces[piece_idx - 1] for piece_idx in line])
        for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
            if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
                return True
        return False

    def _check_2x2_subgrid_win(self):
        for r in range(BOARD_ROWS - 1):
            for c in range(BOARD_COLS - 1):
                subgrid = [self.board[r][c], self.board[r][c+1], self.board[r+1][c], self.board[r+1][c+1]]
                if 0 not in subgrid:  # All cells must be filled
                    characteristics = [pieces[idx - 1] for idx in subgrid]
                    for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
                        if len(set(char[i] for char in characteristics)) == 1:  # All share the same characteristic
                            return True
        return False

    def is_full(self):
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.board[row][col] == 0:
                    return False
        return True
    
    def generate_next_states(self):
        states = []
        for piece in self.available_pieces:
            for row in range(4):
                for col in range(4):
                    if self.board[row][col] == 0:
                        board = deepcopy(self.game_state.board)
                        board[row][col] = self.selected_piece
                        selected_position = (row, col)
                        selected_piece = piece
                        available_pieces = deepcopy(self.available_pieces).remove(piece)
                        states.append(GameState(board, selected_position, selected_piece, available_pieces))
        return states

class Node:
    def __init__(self, game_state, sum_of_reward, parent):
        self.game_state :GameState = game_state
        self.sum_of_reward :int = sum_of_reward
        self.parent :Node = parent
        self.children :list[Node] = []
        self.visit_num :int = 0
    
    def is_leaf(self):
        return self.children == []
    
    def get_max_uct_child(self):
        max_uct = 0
        max_uct_child = None
        for child in self.children:
            uct = child.get_uct()
            if uct > max_uct:
                uct = max_uct
                max_uct_child = child
        return max_uct_child
    
    def get_uct(self):
        if self.visit_num == 0:
            return float("inf")
        return self.sum_of_reward / self.visit_num + CONSTANT * sqrt(log(self.parent.visit_num)/self.visit_num)

    def is_first_visited(self):
        return self.visit_num == 0

    def expand(self):
        if self.children == []:
            self.children += self.generate_children()

    def generate_children(self):
        children = []
        states = self.game_state.generate_next_states()
        for state in states:
            children.append(Node(state, None, self))
        return children

    def is_terminal(self):
        # if self.game_state.is_finished():
        if self.game_state.check_wins():
            return True
        if self.game_state.is_full():
            return True
        return False
    
    def get_result(self):
        if self.game_state.check_wins():
            return 1
        else:
            return 0.5
        
    def update(self, reward):
        self.sum_of_reward += reward
        self.visit_num += 1

    def get_position(self):
        return self.game_state.position

def mcts():
    game_state = GameState(board, None, selected_piece, available_pieces)
    tree_root = Node(game_state, 0, None, 0)

    for _ in range(SEARCH_ITER):
        node = select()
        result = simulate(node)
        backpropagate(node, result) 
    return tree_root.get_max_uct_child()

def select():
    node = tree_root
    while not node.is_leaf():
        node = node.get_max_uct_child()

    if node.is_first_visited():
        return node
    
    if node.is_terminal():
        return node

    node.expand()
    return node.get_max_uct_child() 
    
def simulate(node:Node):
    while not node.is_terminal():
        children = node.generate_children()
        node = random.choice(children)
    return node.get_result()

def backpropagate(selected_node:Node, result):
    node = selected_node 
    while node != None:
        node.update(result)
        node = node.parent
        result = 1 - result

tree_root :Node = None
SEARCH_ITER :int = 10
CONSTANT :float = 1 / sqrt(2)
BOARD_COLS = 4
BOARD_ROWS = 4
pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
    
    def select_piece(self):
        '''
        선택할 말에 해당하는 값을 반환. 예:(0, 1, 0, 1)
        '''
        # Make your own algorithm here
        

        # time.sleep(0.5) # Check time consumption (Delete when you make your algorithm)

        return random.choice(self.available_pieces)

    def place_piece(self, selected_piece): # selected_piece: The selected piece that you have to place on the board (e.g. (1, 0, 1, 0)).
        '''
        말을 놓을 위치 (row, col)을 반환
        '''

        selected_node = mcts()
        return selected_node.get_position()