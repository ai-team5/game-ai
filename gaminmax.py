import numpy as np
import random
from itertools import product
import json
import os
import time


class GeneticMinimaxPlayer():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces
        # GA 파라미터 최적화
        self.population_size = 20  # 증가
        self.generations = 10  # 증가
        self.mutation_rate = 0.15
        self.tournament_size = 5  # 증가
        self.max_depth = 4  # 탐색 깊이 증가
        self.memo_file = "gaminmax_memo.json"
        self.memoization = self.load_memoization()

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

    def evaluate_board(self, board, piece):
        board_key = str(board.tolist())
        piece_key = str(piece)
        memo_key = f"{board_key}_{piece_key}_eval"

        if memo_key in self.memoization:
            return self.memoization[memo_key]

        score = 0

        def check_characteristic_line(line):
            if 0 in line:
                pieces_in_line = [self.pieces[idx - 1] for idx in line if idx != 0]
                if not pieces_in_line:
                    return 0

                # 부분적으로 형성된 패턴 감지
                for char_idx in range(4):
                    chars = [p[char_idx] for p in pieces_in_line]
                    if len(set(chars)) == 1:  # 같은 특성을 가진 경우
                        if len(chars) == 3:  # 승리에 가까운 상황
                            return 80
                        elif len(chars) == 2:  # 잠재적 위험/기회
                            return 30
                return 5  # 기본 점수

            # 완성된 라인 체크
            pieces_in_line = [self.pieces[idx - 1] for idx in line if idx != 0]
            for char_idx in range(4):
                if len(set(p[char_idx] for p in pieces_in_line)) == 1:
                    return 1000  # 승리 상황
            return 10

        # 가로, 세로 평가
        for i in range(4):
            score += check_characteristic_line(board[i])  # 가로
            score += check_characteristic_line(board[:, i])  # 세로

        # 대각선 평가
        diag1 = [board[i][i] for i in range(4)]
        diag2 = [board[i][3 - i] for i in range(4)]
        score += check_characteristic_line(diag1)
        score += check_characteristic_line(diag2)

        # 2x2 서브그리드 평가
        for i in range(3):
            for j in range(3):
                subgrid = [
                    board[i][j], board[i][j + 1],
                    board[i + 1][j], board[i + 1][j + 1]
                ]
                score += check_characteristic_line(subgrid)

        # 중앙 위치 선호
        center_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for pos in center_positions:
            if board[pos[0]][pos[1]] == self.pieces.index(piece) + 1:
                score += 15

        self.memoization[memo_key] = score
        return score

    def tournament_selection(self, population, scores):
        tournament = random.sample(list(zip(population, scores)),
                                   min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: x[1])[0]

    def crossover(self, parent1, parent2):
        if not isinstance(parent1, list):
            parent1, parent2 = list(parent1), list(parent2)
        # 2점 교차로 변경
        points = sorted(random.sample(range(len(parent1)), 2))
        child = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
        return tuple(child)

    def mutate(self, piece):
        if random.random() < self.mutation_rate:
            piece = list(piece)
            # 여러 비트를 변경할 수 있도록 수정
            num_mutations = random.randint(1, 2)
            for _ in range(num_mutations):
                idx = random.randint(0, 3)
                piece[idx] = 1 - piece[idx]
            return tuple(piece)
        return piece

    def minimax(self, board, piece, depth, is_maximizing, alpha, beta):
        board_key = str(board.tolist())
        piece_key = str(piece)
        memo_key = f"{board_key}_{piece_key}_{depth}_{is_maximizing}"

        if memo_key in self.memoization:
            return self.memoization[memo_key]

        if depth == 0:
            result = self.evaluate_board(board, piece)
            self.memoization[memo_key] = result
            return result

        if is_maximizing:
            max_eval = float('-inf')
            available_locs = [(r, c) for r, c in product(range(4), range(4))
                              if board[r][c] == 0]

            for loc in available_locs:
                board[loc[0]][loc[1]] = self.pieces.index(piece) + 1
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

    def select_piece(self):
        board_key = str(self.board.tolist())
        memo_key = f"{board_key}_select_piece"

        if memo_key in self.memoization:
            piece_str = self.memoization[memo_key]
            piece = eval(piece_str)
            if piece in self.available_pieces:
                return piece

        population = random.sample(self.available_pieces,
                                   min(self.population_size, len(self.available_pieces)))

        best_overall_piece = None
        best_overall_score = float('-inf')

        for generation in range(self.generations):
            scores = []
            for piece in population:
                score = self.minimax(self.board.copy(), piece, self.max_depth, True,
                                     float('-inf'), float('inf'))
                scores.append(score)

                # 현재까지의 최고 점수 업데이트
                if score > best_overall_score and piece in self.available_pieces:
                    best_overall_score = score
                    best_overall_piece = piece

            new_population = []
            elite_count = 2  # 엘리트 보존
            elite_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:elite_count]
            new_population.extend([population[i] for i in elite_indices])

            while len(new_population) < len(population):
                parent1 = self.tournament_selection(population, scores)
                parent2 = self.tournament_selection(population, scores)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                if child in self.available_pieces:
                    new_population.append(child)

            if new_population:
                population = new_population

        selected_piece = best_overall_piece if best_overall_piece else random.choice(self.available_pieces)
        self.memoization[memo_key] = str(selected_piece)
        return selected_piece

    def place_piece(self):
        board_key = str(self.board.tolist())
        memo_key = f"{board_key}_place_piece"

        if memo_key in self.memoization:
            loc_str = self.memoization[memo_key]
            return eval(loc_str)

        available_locs = [(row, col) for row, col in product(range(4), range(4))
                          if self.board[row][col] == 0]

        best_location = None
        best_score = float('-inf')

        for loc in available_locs:
            self.board[loc[0]][loc[1]] = 1
            score = self.minimax(self.board.copy(), (0, 0, 0, 0), self.max_depth, False,
                                 float('-inf'), float('inf'))
            self.board[loc[0]][loc[1]] = 0

            if score > best_score:
                best_score = score
                best_location = loc

        selected_location = best_location if best_location else random.choice(available_locs)
        self.memoization[memo_key] = str(selected_location)
        return selected_location