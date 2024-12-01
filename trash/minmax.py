import random 
import numpy as np
from itertools import product
import json
import os
from functools import lru_cache


class MinimaxPlayer:
    def __init__(self, board, available_pieces, current_piece=None, max_depth=4):
        self.depth_limit = max_depth
        self.depth_limit_case_count = 0
        self.pieces = tuple((i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2))
        self._center_positions = ((1,1), (1,2), (2,1), (2,2))
        self._corner_positions = ((0,0), (0,3), (3,0), (3,3))
        self._edge_positions = ((0,1), (0,2), (1,0), (1,3), (2,0), (2,3), (3,1), (3,2))
        self.memo_file = "minimax_memo.json"
        self.name = "donghyun minmax"
        self.memoization = self.load_memoization()
        self.board = board
        self.available_pieces = tuple(available_pieces)
        self.max_depth = max_depth
        self.current_piece = current_piece if current_piece else self.select_piece()

    def load_memoization(self):
        try:
            with open(self.memo_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Failed to load {self.memo_file}")
            return {}

    def save_memoization(self):
        try:
            with open(self.memo_file, 'w') as f:
                json.dump(self.memoization, f)
        except:
            pass

    def __del__(self):
        self.save_memoization()

    def select_piece(self):
        placed_piece_count = 0
        for row in range(4):
            for col in range(4):
                if self.board[row][col] != 0:
                    placed_piece_count += 1

        if placed_piece_count <= 1:
            return random.choice(self.available_pieces)
        else:
            return self.find_most_disadvantageous_piece()

    def place_piece(self, selected_piece):
        placed_piece_count = 0
        for row in range(4):
            for col in range(4):
                if self.board[row][col] != 0:
                    placed_piece_count += 1

        if placed_piece_count <= 1:
            available_locs = [(row, col) for row, col in product(range(1,3), range(1,3)) if self.board[row][col]==0]
            return random.choice(available_locs)

        if placed_piece_count >= 5:
            self.depth_limit = 2
        elif placed_piece_count >= 8:
            self.depth_limit = 3
        elif placed_piece_count >= 10:
            self.depth_limit = 5

        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        best_score = float('-inf')
        best_location = None
        next_piece = None

        board_copy = np.array(self.board.copy())
        available_pieces = list(self.available_pieces)
        available_pieces.remove(selected_piece)

        for row, col in available_locs:
            for piece in available_pieces:
                board_copy[row][col] = self.piece_to_idx(selected_piece)
                score = self.minimax(board_copy, False, piece, self.depth_limit, available_pieces)
                if score > best_score:
                    best_score = score
                    best_location = (row, col)
                    next_piece = piece
                board_copy[row][col] = 0

        return best_location or random.choice(available_locs)

    def minimax(self, board, is_maximizing, selected_piece, depth, available_pieces, alpha=float('-inf'), beta=float('inf')):
        if self.evaluate_win(board):
            return -100 if is_maximizing else 100
            
        if self.is_full(board):
            return 0
            
        if depth == 0:
            self.depth_limit_case_count += 1
            return self.evaluate_board(board)

        available_pieces = available_pieces[:]
        available_pieces.remove(selected_piece)

        if is_maximizing:
            best_score = float('-inf')
            for row in range(4):
                for col in range(4):
                    for piece in available_pieces:
                        if board[row][col] == 0:
                            board[row][col] = self.piece_to_idx(selected_piece)
                            score = self.minimax(board, False, piece, depth-1, available_pieces, alpha, beta)
                            best_score = max(best_score, score)
                            alpha = max(alpha, best_score)
                            board[row][col] = 0
                            if beta <= alpha:
                                return best_score
            return best_score
        else:
            best_score = float('inf')
            for row in range(4):
                for col in range(4):
                    for piece in available_pieces:
                        if board[row][col] == 0:
                            board[row][col] = self.piece_to_idx(selected_piece)
                            score = self.minimax(board, True, piece, depth-1, available_pieces, alpha, beta)
                            best_score = min(best_score, score)
                            beta = min(beta, best_score)
                            board[row][col] = 0
                            if beta <= alpha:
                                return best_score
            return best_score

    def find_most_disadvantageous_piece(self):
        worst_piece = None
        worst_value = float('inf')

        board_copy = np.array(self.board.copy())
        for piece in self.available_pieces:
            value = self.minimax(board_copy, True, piece, self.depth_limit, list(self.available_pieces))
            if value < worst_value:
                worst_value = value
                worst_piece = piece

        return worst_piece or random.choice(self.available_pieces)

    def is_full(self, board):
        return not any(0 in row for row in board)

    def piece_to_idx(self, piece):
        if piece is None:
            raise ValueError("Cannot convert None piece to index")
        return self.pieces.index(piece) + 1

    def evaluate_win(self, board):
        for i in range(4):
            if self.check_line([board[i][j] for j in range(4)]) or \
               self.check_line([board[j][i] for j in range(4)]):
                return True

        if self.check_line([board[i][i] for i in range(4)]) or \
           self.check_line([board[i][3-i] for i in range(4)]):
            return True

        if self.check_2x2_subgrid_win(board):
            return True

        return False

    def check_line(self, line):
        if 0 in line:
            return False
        characteristics = np.array([self.pieces[piece_idx-1] for piece_idx in line])
        return any(len(set(characteristics[:,i])) == 1 for i in range(4))

    def check_2x2_subgrid_win(self, board):
        for r in range(3):
            for c in range(3):
                subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                if 0 not in subgrid:
                    characteristics = [self.pieces[idx-1] for idx in subgrid]
                    if any(len(set(char[i] for char in characteristics)) == 1 for i in range(4)):
                        return True
        return False

    def evaluate_board(self, board):
        score = 50
        for row in range(4):
            for col in range(4):
                if board[row][col] != 0:
                    if (row,col) in self._center_positions:
                        score += 200
                    elif (row,col) in self._corner_positions:
                        score += 100
                    elif (row,col) in self._edge_positions:
                        score += 40
        return score
