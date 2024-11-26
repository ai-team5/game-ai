import numpy as np
import random
from itertools import product
from copy import deepcopy
from math import sqrt, log
import time
import pickle
import os

tree_root = None
RAND_PLACE_NUM = 1
ITER_NUM = 3000
CONSTANT = 1
BOARD_COLS = 4
BOARD_ROWS = 4
pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces

MEMOIZATION_FILE = 'mcts_memo.pkl'

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
        self.load_memoization()
    
    def load_memoization(self):
        global tree_root
        if os.path.exists(MEMOIZATION_FILE):
            try:
                with open(MEMOIZATION_FILE, 'rb') as f:
                    tree_root = pickle.load(f)
                print("Loaded memoized MCTS tree")
            except:
                print("Failed to load memoization file")
                tree_root = None
    
    def save_memoization(self):
        global tree_root
        try:
            with open(MEMOIZATION_FILE, 'wb') as f:
                pickle.dump(tree_root, f)
            print("Saved MCTS tree to file")
        except:
            print("Failed to save memoization file")

    def select_piece(self):
        '''
        선택할 말에 해당하는 값을 반환. 예:(0, 1, 0, 1)
        '''
        # Make your own algorithm here

 
        # return random.choice(self.available_pieces)

        placed_piece_count = 0
        for row in range(4):
            for col in range(4):
                if self.board[row][col] != 0:
                    placed_piece_count += 1

        if placed_piece_count <= RAND_PLACE_NUM + 1:
            return random.choice(self.available_pieces)
    
        global tree_root
        selected_node = tree_root.get_best_child()

        # print("debug : promising node")
        # selected_node.print()
        self.save_memoization()

        return selected_node.get_selected_piece()

    def place_piece(self, selected_piece): # selected_piece: The selected piece that you have to place on the board (e.g. (1, 0, 1, 0)).
        '''
        말을 놓을 위치 (row, col)을 반환
        '''

        placed_piece_count = 0
        for row in range(4):
            for col in range(4):
                if self.board[row][col] != 0:
                    placed_piece_count += 1
        
        if placed_piece_count <= RAND_PLACE_NUM:
            # Available locations to place the piece
            available_locs = [(row, col) for row, col in product(range(1,3), range(1,3)) if self.board[row][col]==0]
            return random.choice(available_locs)

        board = deepcopy(self.board)
        available_pieces = deepcopy(self.available_pieces)
        available_pieces.remove(selected_piece)
        selected_node = mcts(board, available_pieces, selected_piece)
        self.save_memoization()
        return selected_node.get_selected_position()
    
class GameState:
    def __init__(self, board, selected_position, selected_piece, available_pieces, player):
        self.board = board
        self.selected_position = selected_position
        self.selected_piece = selected_piece
        self.available_pieces = available_pieces
        self.player = player

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
    
    def next_states(self):
        # print(f"debug : board = {self.board}")
        states = []
        if self.available_pieces == []:
            board = deepcopy(self.board)
            for row in range(4):
                for col in range(4):
                    if self.board[row][col] == 0:
                        board[row][col] = pieces.index(self.selected_piece) + 1
            player = 3 - self.player
            states.append(GameState(board, None, None, [], player))
            return states
        
        for row in range(4):
            for col in range(4):
                for piece in self.available_pieces:
                    if self.board[row][col] == 0:
                        board = deepcopy(self.board)
                        board[row][col] = pieces.index(self.selected_piece) + 1
                        selected_position = (row, col)
                        available_pieces = deepcopy(self.available_pieces)
                        available_pieces.remove(piece)
                        player = 3 - self.player
                        states.append(GameState(board, selected_position, piece, available_pieces, player))
        # print("debug : ")
        # print(f"board = {self.board}")
        # print(f"board = {self.available_pieces}")
        assert states != []
            
        return states

class Node:
    def __init__(self, game_state, sum_of_reward, parent):
        self.game_state:GameState = game_state
        self.sum_of_reward:int = sum_of_reward
        self.parent:Node = parent
        self.children:list = []
        self.visit_num:int  = 0
    
    def is_leaf(self):
        return self.children == []
    
    def get_best_child(self):
        max_uct = 0
        max_uct_child = None
        for child in self.children:
            uct = child.get_uct()
            if uct > max_uct:
                max_uct = uct
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
            self.children = self.next_nodes()

    def next_nodes(self):
        children = []
        for state in self.game_state.next_states():
            children.append(Node(state, 0, self))
        return children

    def is_terminal(self):
        if self.game_state.is_finished():
            return True
        return False
    
    def get_result(self):
        if self.game_state.check_wins():
            return {"reward" : 1, "player": self.game_state.player }
        return {"reward" : 0.2, "player": self.game_state.player }
            
    def update(self, result):
        assert result is not None
        if self.game_state.player == result["player"]:
            self.sum_of_reward += result["reward"]
        self.visit_num += 1

    def get_selected_position(self):
        return self.game_state.selected_position
    
    def get_selected_piece(self):
        return self.game_state.selected_piece
    
    def print(self):
        print(f"board = {self.game_state.board}")
        print(f"selected_piece = {self.game_state.selected_piece}")
        print(f"uct = {self.get_uct()}")
        print(f"visit = {self.visit_num}")
        print("---------------")


def mcts(board, available_pieces, selected_piece):
    global tree_root
    player = 2    

    if tree_root != None:
        new_root = None 
        for child in tree_root.children:
            if np.array_equal(child.game_state.board, board) and child.game_state.selected_piece == selected_piece:
                print(np.array_equal(child.game_state.board, board))
                new_root = child 
                break

        if new_root != None:
            tree_root = new_root
        else:
            tree_root = None
        

    if tree_root == None:
        game_state = GameState(board, None, selected_piece, available_pieces, player)
        tree_root = Node(game_state, 0, None)

    for iter in range(ITER_NUM):
        print(f"debug : mcts iter = {iter}")
        node = select(tree_root)

        # print("debug : print node after selection")
        # node.print()

        result = simulate(node)
        backpropagate(node, result) 

    # print("debug : print root's children")
    # for i, child in enumerate(tree_root.children):
    #     print(f"child {i}")
    #     child.print()

    child = tree_root.get_best_child()
    print("debug : promising node")
    child.print()

    return child 

def select(node:Node):
    while not node.is_leaf():
        node = node.get_best_child()

    if node.is_first_visited() or node.is_terminal():
        return node

    node.expand()
    return node.get_best_child()
    
def simulate(node:Node):
    while not node.is_terminal():
        children = node.next_nodes()
        # print("debug : game_state.board",node.game_state.board)
        # print("debug : ", children)
        node = random.choice(children)
    return node.get_result()

def backpropagate(node:Node, result:dict):
    while node != None:
        node.update(result)
        node = node.parent
