import numpy as np
import random
from itertools import product
from copy import deepcopy
import time

PLACED_PIECE_NUM = 1
BOARD_ROWS = 4
BOARD_COLS = 4
MAX_WIN_SCORE = 100
MIN_WIN_SCORE = -100
DRAW_SCORE = 0

pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
next_piece = None

class P2():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
        self.name = "ikhwan minmax"
        self.depth_limit = 1
        self.depth_limit_case_count = 0

    def select_piece(self):
        '''
        선택할 말에 해당하는 값을 반환. 예:(0, 1, 0, 1)
        '''
        # Make your own algorithm here
        placed_piece_count = 0
        for row in range(4):
            for col in range(4):
                if self.board[row][col] != 0:
                    placed_piece_count += 1

        if placed_piece_count <= PLACED_PIECE_NUM + 1:
            return random.choice(self.available_pieces)
        else:   
            return next_piece

    def place_piece(self, selected_piece): # selected_piece: The selected piece that you have to place on the board (e.g. (1, 0, 1, 0)).
        '''
        말을 놓을 위치 (row, col)을 반환
        '''
        # global DEPTH
        # DEPTH = 1
        # Make your own algorithm here
        placed_piece_count = 0
        for row in range(4):
            for col in range(4):
                if self.board[row][col] != 0:
                    placed_piece_count += 1
        
        print(f"debug : place_piece (), placed_piece_count {placed_piece_count}")

        if placed_piece_count <= PLACED_PIECE_NUM:
            # Available locations to place the piece
            available_locs = [(row, col) for row, col in product(range(1,3), range(1,3)) if self.board[row][col]==0]
            return random.choice(available_locs)
        
        if placed_piece_count >= 5:
            self.depth_limit = 2
        elif placed_piece_count >= 8:
            self.depth_limit = 3
        elif placed_piece_count >= 10:
            self.depth_limit = 5
        
        global next_piece
        board = np.array(deepcopy(self.board.tolist()))
        available_pieces = deepcopy(self.available_pieces)
    
        available_pieces.remove(selected_piece)
        assert len(available_pieces) != 0
        best_score = -1e9
        next_location = None 
        for row in range(4):
            for col in range(4):
                for piece in available_pieces:
                    if board[row][col] == 0:
                        board[row][col] = pieces.index(selected_piece) + 1
                        print(available_pieces)
                        score = self.minmax(board, False, piece, self.depth_limit, available_pieces)                       
                        if best_score < score:
                            best_score = score
                            next_location = (row, col)
                            next_piece = piece
                        board[row][col] = 0
                        
        # print(f"debug : place_piece() :: board {self.board}")
        print(f"debug : place_piece() :: next_location {next_location}, next_piece {next_piece}")
        
        return next_location
    
    def minmax(self, board, is_maximizing:bool, selected_piece:int, depth:int, available_pieces, alpha=-1e9, beta=1e9):
        board = deepcopy(board)
        available_pieces = available_pieces[:]
        available_pieces.remove(selected_piece)
        # assert len(available_pieces) != 0

        if self.check_win(board):
            if is_maximizing:
                return MIN_WIN_SCORE
            return MAX_WIN_SCORE
        
        if self.check_board_is_full(board):
            return DRAW_SCORE
        
        if depth == 0:
            # global depth_limit_case_count
            self.depth_limit_case_count += 1
            print(f"debug: minmax :: p2 depth limit {self.depth_limit} case = {self.depth_limit_case_count}")
            return self.evaluate(board)

        if is_maximizing:
            best_score = -1e9 
            for row in range(4):
                for col in range(4):
                    for piece in available_pieces:
                        if board[row][col] == 0:
                            board[row][col] = pieces.index(selected_piece) + 1
                            score = self.minmax(board, False, piece, depth - 1, available_pieces, alpha, beta)
                            best_score = max(best_score, score)
                            alpha = max(best_score, alpha)
                            if beta < alpha:
                                return best_score
                            board[row][col] = 0  
            return best_score
        else:
            best_score = 1e9 
            for row in range(4):
                for col in range(4):
                    for piece in available_pieces:
                        if board[row][col] == 0:
                            board[row][col] = pieces.index(selected_piece) + 1
                            score = self.minmax(board, True, piece, depth - 1, available_pieces, alpha, beta)
                            best_score = min(best_score, score)
                            beta = min(best_score, beta)
                            if alpha > beta:
                                return best_score
                            board[row][col] = 0  
            return best_score

    def evaluate(self, board):
        # 내가 이길 수 있도록 놓을 수 있는 선이 가장 많은 게 높은 점수를 받아야 할 것 같음
        
        return 50

    def check_win(self, board):
        # Check rows, columns, and diagonals
        for col in range(BOARD_COLS):
            if self.check_line([board[row][col] for row in range(BOARD_ROWS)]):
                return True
        
        for row in range(BOARD_ROWS):
            if self.check_line([board[row][col] for col in range(BOARD_COLS)]):
                return True
            
        if self.check_line([board[i][i] for i in range(BOARD_ROWS)]) or self.check_line([board[i][BOARD_ROWS - i - 1] for i in range(BOARD_ROWS)]):
            return True

        # Check 2x2 sub-grids
        if self.check_2x2_subgrid_win(board):
            return True
        
        return False

    def check_line(self, line):
        if 0 in line:
            return False  # Incomplete line
        characteristics = np.array([pieces[piece_idx - 1] for piece_idx in line])
        for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
            if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
                return True
        return False

    def check_2x2_subgrid_win(self, board):
        for r in range(BOARD_ROWS - 1):
            for c in range(BOARD_COLS - 1):
                subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                if 0 not in subgrid:  # All cells must be filled
                    characteristics = [pieces[idx - 1] for idx in subgrid]
                    for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
                        if len(set(char[i] for char in characteristics)) == 1:  # All share the same characteristic
                            return True
        return False

    def check_board_is_full(self, board):
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if board[row][col] == 0:
                    return False
        return True





        
                    

                
            
            


        
            
