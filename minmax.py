import random 
import numpy as np
from itertools import product
import json
import os


class MinimaxPlayer:
    def __init__(self, board, available_pieces, current_piece=None, max_depth=3):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.memo_file = "minimax_memo.json"
        self.memoization = self.load_memoization()
        self.board = board
        self.available_pieces = available_pieces
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
        if self.is_empty_board():
            # 첫 턴에는 중앙에 놓을 수 있는 가장 유리한 피스 선택
            best_piece = None
            best_value = float('-inf')
            for piece in self.available_pieces:
                self.board[1][1] = self.piece_to_idx(piece)
                value = self.minimax(self.board.copy(), piece, 2, False, float('-inf'), float('inf'))
                self.board[1][1] = 0
                if value > best_value:
                    best_value = value
                    best_piece = piece
            return best_piece or random.choice(self.available_pieces)
            
        return self.find_most_disadvantageous_piece()

    def place_piece(self):
        if self.current_piece is None:
            self.current_piece = self.select_piece()
            if self.current_piece is None:
                raise ValueError("No pieces available to select")
            
        if self.is_empty_board():
            return (1, 1)
        
        available_locs = [(row, col) for row, col in product(range(4), range(4))
                          if self.board[row][col] == 0]

        best_location = None
        best_score = float('-inf')

        board_key = str(self.board.tolist())
        piece_key = str(self.current_piece)
        memo_key = f"{board_key}_{piece_key}"

        if memo_key in self.memoization:
            scores = self.memoization[memo_key]
            for loc in available_locs:
                loc_key = str(loc)
                if loc_key in scores:
                    score = scores[loc_key]
                    if score > best_score:
                        best_score = score
                        best_location = loc
            if best_location:
                return best_location

        scores = {}
        for loc in available_locs:
            self.board[loc[0]][loc[1]] = self.piece_to_idx(self.current_piece)
            score = self.minimax(self.board.copy(), self.current_piece, self.max_depth, False,
                                 float('-inf'), float('inf'))
            self.board[loc[0]][loc[1]] = 0
            scores[str(loc)] = score

            if score > best_score:
                best_score = score
                best_location = loc

        self.memoization[memo_key] = scores
        return best_location if best_location else random.choice(available_locs)

    def minimax(self, board, piece, depth, is_maximizing, alpha, beta):
        board_key = str(board.tolist())
        piece_key = str(piece)
        memo_key = f"{board_key}_{piece_key}_{depth}_{is_maximizing}"

        if memo_key in self.memoization:
            return self.memoization[memo_key]

        if depth == 0 or self.evaluate_win(board) or self.is_full(board):
            result = self.evaluate_board(board, piece)
            self.memoization[memo_key] = result
            return result

        if is_maximizing:
            max_eval = float('-inf')
            available_locs = [(r, c) for r, c in product(range(4), range(4))
                              if board[r][c] == 0]

            for loc in available_locs:
                board[loc[0]][loc[1]] = self.piece_to_idx(piece)
                eval = self.minimax(board, piece, depth - 1, False, alpha, beta)
                board[loc[0]][loc[1]] = 0
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.memoization[memo_key] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            available_locs = [(r, c) for r, c in product(range(4), range(4))
                              if board[r][c] == 0]

            for loc in available_locs:
                board[loc[0]][loc[1]] = 0
                eval = self.minimax(board, piece, depth - 1, True, alpha, beta)
                board[loc[0]][loc[1]] = 0
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.memoization[memo_key] = min_eval
            return min_eval

    def find_most_disadvantageous_piece(self):
        worst_piece = None
        worst_value = float('inf')

        board_key = str(self.board.tolist())
        memo_key = f"{board_key}_disadvantageous"

        if memo_key in self.memoization:
            piece_str = self.memoization[memo_key]
            return eval(piece_str) if piece_str else random.choice(self.available_pieces)

        for piece in self.available_pieces:
            value = self.minimax(self.board.copy(), piece, self.max_depth, True,
                                float('-inf'), float('inf'))
            if value < worst_value:
                worst_value = value
                worst_piece = piece

        self.memoization[memo_key] = str(worst_piece) if worst_piece else ""
        return worst_piece or random.choice(self.available_pieces)

    def is_empty_board(self):
        return not (self.board != 0).any()

    def is_full(self, board):
        return (board != 0).all()

    def piece_to_idx(self, piece):
        if piece is None:
            raise ValueError("Cannot convert None piece to index")
        return self.pieces.index(piece) + 1

    def evaluate_win(self, board):
        # Check rows
        for row in board:
            if len(set(row)) == 1 and row[0] != 0:
                return True

        # Check columns 
        for col in board.T:
            if len(set(col)) == 1 and col[0] != 0:
                return True

        # Check diagonals
        diag1 = [board[i][i] for i in range(4)]
        diag2 = [board[i][3-i] for i in range(4)]
        if len(set(diag1)) == 1 and diag1[0] != 0:
            return True
        if len(set(diag2)) == 1 and diag2[0] != 0:
            return True

        # Check 2x2 subgrids
        for i in range(3):
            for j in range(3):
                subgrid = [board[i][j], board[i][j+1], board[i+1][j], board[i+1][j+1]]
                if 0 not in subgrid and len(set(subgrid)) == 1:
                    return True

        return False

    def evaluate_board(self, board, piece):
        board_key = str(board.tolist())
        piece_key = str(piece)
        memo_key = f"{board_key}_{piece_key}_eval"

        if memo_key in self.memoization:
            return self.memoization[memo_key]

        score = 0
        center_positions = [(1,1), (1,2), (2,1), (2,2)]
        
        def check_characteristic_line(line):
            if 0 in line:
                pieces_in_line = [self.pieces[idx - 1] for idx in line if idx != 0]
                if not pieces_in_line:
                    return 0

                for char_idx in range(4):
                    chars = [p[char_idx] for p in pieces_in_line]
                    if len(set(chars)) == 1:
                        if len(chars) == 3:
                            return 80
                        elif len(chars) == 2:
                            return 30
                return 5

            pieces_in_line = [self.pieces[idx - 1] for idx in line if idx != 0]
            for char_idx in range(4):
                if len(set(p[char_idx] for p in pieces_in_line)) == 1:
                    return 1000
            return 10

        # 가로, 세로 평가
        for i in range(4):
            score += check_characteristic_line(board[i])
            score += check_characteristic_line(board[:, i])

        # 대각선 평가
        diag1 = [board[i][i] for i in range(4)]
        diag2 = [board[i][3 - i] for i in range(4)]
        score += check_characteristic_line(diag1)
        score += check_characteristic_line(diag2)

        # 중앙 위치 선호
        for pos in center_positions:
            if board[pos[0]][pos[1]] == self.piece_to_idx(piece):
                score += 15

        self.memoization[memo_key] = score
        return score
