import numpy as np
import random
from itertools import product
import json
import os
import time
from copy import deepcopy

# 상수 정의
MAX_WIN_SCORE = 1000000
MIN_WIN_SCORE = -1000000 
DRAW_SCORE = 0
TIMEOUT_SECONDS = 5  # 최대 실행 시간 제한

class GeneticMinimaxPlayer():
    def __init__(self, board, available_pieces):
        self.pieces = tuple((i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2))
        self.board = board
        self.name = "gaminmax"
        self.available_pieces = list(available_pieces)
        # 성능과 시간을 고려한 파라미터 조정
        self.population_size = 30  # population 크기 감소
        self.generations = 15  # 세대수 감소
        self.mutation_rate = 0.2
        self.tournament_size = 8  # tournament 크기 감소
        self.max_depth = 4  # 탐색 깊이 감소
        self.memo_file = "gaminmax_memo.json"
        self.memoization = self.load_memoization()
        self._center_positions = ((1,1), (1,2), (2,1), (2,2))
        self._corner_positions = ((0,0), (0,3), (3,0), (3,3))
        self._edge_positions = ((0,1), (0,2), (1,0), (1,3), (2,0), (2,3), (3,1), (3,2))
        self.start_time = 0

    def load_memoization(self):
        try:
            if os.path.exists(self.memo_file):
                with open(self.memo_file, 'r') as f:
                    return json.load(f)
            return {}
        except:
            return {}

    def save_memoization(self):
        try:
            with open(self.memo_file, 'w') as f:
                json.dump(self.memoization, f)
        except:
            pass

    def __del__(self):
        self.save_memoization()

    def check_timeout(self):
        if time.time() - self.start_time > TIMEOUT_SECONDS:
            raise TimeoutError("Computation time exceeded")

    def evaluate_board(self, board, piece):
        self.check_timeout()
        board_key = str(board)
        piece_key = str(piece)
        memo_key = f"{board_key}_{piece_key}_eval"

        if memo_key in self.memoization:
            return self.memoization[memo_key]

        score = 0
        board_array = np.array(board)

        def check_characteristic_line(line):
            if 0 in line:
                pieces_in_line = tuple(self.pieces[idx - 1] for idx in line if idx != 0)
                if not pieces_in_line:
                    return 0

                for char_idx in range(4):
                    chars = tuple(p[char_idx] for p in pieces_in_line)
                    chars_set = set(chars)
                    if len(chars_set) == 1:
                        if len(chars) == 3:
                            return 500
                        elif len(chars) == 2:
                            return 200
                        else:
                            return 50

            pieces_in_line = tuple(self.pieces[idx - 1] for idx in line if idx != 0)
            for char_idx in range(4):
                if len(set(p[char_idx] for p in pieces_in_line)) == 1:
                    return 10000

            return 20

        # 기본 평가
        for i in range(4):
            score += check_characteristic_line(tuple(board_array[i])) 
            score += check_characteristic_line(tuple(board_array[:,i]))

        # 위치 보너스
        piece_idx = self.pieces.index(piece) + 1
        score += sum(100 for pos in self._center_positions if board_array[pos[0]][pos[1]] == piece_idx)
        score += sum(50 for pos in self._corner_positions if board_array[pos[0]][pos[1]] == piece_idx)
        score += sum(20 for pos in self._edge_positions if board_array[pos[0]][pos[1]] == piece_idx)

        self.memoization[memo_key] = score
        return score    

    def check_win(self, board):
        self.check_timeout()
        for i in range(4):
            row = board[i]
            col = board[:,i]
            if row[0] != 0 and len(set(row)) == 1:
                return True
            if col[0] != 0 and len(set(col)) == 1:
                return True
                
        diag1 = board.diagonal()
        diag2 = np.fliplr(board).diagonal()
        
        if diag1[0] != 0 and len(set(diag1)) == 1:
            return True
        if diag2[0] != 0 and len(set(diag2)) == 1:
            return True

        for r in range(3):
            for c in range(3):
                subgrid = board[r:r+2, c:c+2].flatten()
                if 0 not in subgrid and len(set(subgrid)) == 1:
                    return True
            
        return False

    def tournament_selection(self, population, scores):
        tournament_size = min(self.tournament_size, len(population))
        tournament = random.sample(list(zip(population, scores)), tournament_size)
        return max(tournament, key=lambda x: x[1])[0]

    def crossover(self, parent1, parent2):
        if not isinstance(parent1, list):
            parent1, parent2 = list(parent1), list(parent2)
        point = random.randint(1, len(parent1)-1)
        child = parent1[:point] + parent2[point:]
        return tuple(child)

    def mutate(self, piece):
        if random.random() < self.mutation_rate:
            piece = list(piece)
            idx = random.randint(0, 3)
            piece[idx] = 1 - piece[idx]
            return tuple(piece)
        return piece

    def minimax(self, board, piece, depth, is_maximizing, alpha, beta):
        self.check_timeout()
        board_array = np.array(board)

        if self.check_win(board_array):
            return MIN_WIN_SCORE if is_maximizing else MAX_WIN_SCORE

        if depth == 0:
            return self.evaluate_board(board, piece)

        available_pieces = [p for p in self.available_pieces if p != piece]
        if not available_pieces:
            return DRAW_SCORE

        empty_cells = [(r,c) for r,c in product(range(4), range(4)) 
                      if board_array[r][c] == 0]

        if is_maximizing:
            best_score = -1e9
            for row,col in empty_cells:
                for next_piece in available_pieces:
                    board_array[row][col] = self.pieces.index(piece) + 1
                    score = self.minimax(tuple(map(tuple, board_array)), next_piece, 
                                     depth-1, False, alpha, beta)
                    board_array[row][col] = 0
                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break
            return best_score
        else:
            best_score = 1e9  
            for row,col in empty_cells:
                for next_piece in available_pieces:
                    board_array[row][col] = self.pieces.index(piece) + 1
                    score = self.minimax(tuple(map(tuple, board_array)), next_piece,
                                     depth-1, True, alpha, beta)
                    board_array[row][col] = 0
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break
            return best_score

    def select_piece(self):
        self.start_time = time.time()
        try:
            board_key = str(tuple(map(tuple, self.board)))
            memo_key = f"{board_key}_select_piece"

            if memo_key in self.memoization:
                piece = eval(self.memoization[memo_key])
                if piece in self.available_pieces:
                    return piece

            best_piece = None
            best_score = float('-inf')

            for piece in self.available_pieces:
                score = self.minimax(tuple(map(tuple, self.board)), piece, 
                                 self.max_depth, True, float('-inf'), float('inf'))
                if score > best_score:
                    best_score = score
                    best_piece = piece

            selected_piece = best_piece if best_piece else self.available_pieces[0]
            self.memoization[memo_key] = str(selected_piece)
            return selected_piece

        except TimeoutError:
            return random.choice(self.available_pieces)

    def place_piece(self, selected_piece):
        self.start_time = time.time()
        try:
            board_key = str(tuple(map(tuple, self.board)))
            memo_key = f"{board_key}_place_piece"

            if memo_key in self.memoization:
                return eval(self.memoization[memo_key])

            available_locs = [(r,c) for r,c in product(range(4), range(4))
                           if self.board[r][c] == 0]

            best_location = None
            best_score = float('-inf')

            for loc in available_locs:
                board_copy = self.board.copy()
                board_copy[loc[0]][loc[1]] = self.pieces.index(selected_piece) + 1
                
                score = self.minimax(tuple(map(tuple, board_copy)), selected_piece,
                                 self.max_depth, False, float('-inf'), float('inf'))
                
                if loc in self._center_positions:
                    score += 100
                elif loc in self._corner_positions:
                    score += 50
                elif loc in self._edge_positions:
                    score += 20
                    
                if score > best_score:
                    best_score = score
                    best_location = loc

            selected_location = best_location if best_location else available_locs[0]
            self.memoization[memo_key] = str(selected_location)
            return selected_location

        except TimeoutError:
            return random.choice(available_locs)
