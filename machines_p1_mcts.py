import numpy as np
import random
from itertools import product
from copy import deepcopy
from math import sqrt, log

class GameState:
    def __init__(self, board, selected_position, selected_piece, next_available_pieces, player):
        self.board :list[list] = board
        self.selected_position :tuple = selected_position # location selected by me
        self.selected_piece :tuple = selected_piece # opposite's piece
        self.next_available_pieces :list[tuple] = next_available_pieces
        self.player :int = player

    def is_finished(self):
        if self.check_wins():
            return True
        if self._is_full():
            return True
        return False
    
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

    def _is_full(self):
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.board[row][col] == 0:
                    return False
        return True
    
    def generate_next_states(self):
        states = []
        for piece in self.next_available_pieces:
            for row in range(4):
                for col in range(4):
                    if self.board[row][col] == 0:
                        board = deepcopy(self.game_state.board)
                        board[row][col] = self.selected_piece
                        selected_position = (row, col)
                        selected_piece = piece
                        available_pieces = deepcopy(self.next_available_pieces).remove(piece)
                        player = 3 - self.player
                        states.append(GameState(board, selected_position, selected_piece, available_pieces, player))
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
        if self.game_state.is_finished():
            return True
        return False
    
    def get_result(self):
        if self.game_state.check_wins():
            return {"reward" : 1, "player": self.game_state.player}
        else:
            return {"reward" : 0.5, "player": self.game_state.player}
            # self.sum_of_reward = 0.5
            
    def update(self, result):
        if self.game_state.player == result["player"]:
            reward = result["reward"] 
        else:
            reward = 1 - result["reward"]
        self.sum_of_reward += reward
        self.visit_num += 1

    def get_position(self):
        return self.game_state.selected_position

def mcts_for_selection():
    pass

def mcts_for_location(board, available_pieces, selected_piece):
    game_state = GameState(board, None, selected_piece, available_pieces, 2)
    tree_root = Node(game_state, None, None)
    for _ in range(SEARCH_ITER):
        node = select(tree_root)
        result = simulate(node)
        backpropagate(node, result) 
    return tree_root.get_max_uct_child()

def select(node:Node):
    while not node.is_leaf():
        node = node.get_max_uct_child()

    if node.is_first_visited():
        return node
    
    if node.is_terminal():
        return node

    node.expand()
    return node.get_max_uct_child() 
    
def simulate(selected_node:Node):
    node = selected_node
    while not node.is_terminal():
        children = node.generate_children()
        node = random.choice(children)
    return node.get_result()

def backpropagate(simulated_node:Node, result:dict):
    node = simulated_node 
    while node != None:
        node.update(result)
        node = node.parent

tree_root :Node = None
SEARCH_ITER = 10
CONSTANT = 1 / sqrt(2)
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

        board = deepcopy(self.board)
        available_pieces = deepcopy(self.available_pieces)
        selected_node = mcts_for_selection(board, available_pieces)

        # time.sleep(0.5) # Check time consumption (Delete when you make your algorithm)
        return selected_node.get_selected_piece()

    def place_piece(self, selected_piece): # selected_piece: The selected piece that you have to place on the board (e.g. (1, 0, 1, 0)).
        '''
        말을 놓을 위치 (row, col)을 반환
        '''
        board = deepcopy(self.board)
        available_pieces = deepcopy(self.available_pieces)
        selected_node = mcts_for_location(board, available_pieces, selected_piece)
        return selected_node.get_position()