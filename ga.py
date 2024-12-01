import random
import numpy as np
import json
import os
import time
from copy import deepcopy

class GeneticAlgorithm:
    def __init__(self, board, available_pieces, current_piece=None, population_size=200, generations=100, mutation_rate=0.2, tournament_size=40):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.memo_file = "ga_memo.json"
        self.name = "ga"
        self.memoization = self.load_memoization()
        self.is_new_memo = not os.path.exists(self.memo_file)
        self.board = board
        self.available_pieces = available_pieces
        self.current_piece = current_piece
        # 더 강력한 파라미터 설정
        self.population_size = population_size  # 더 큰 population으로 다양성 극대화 
        self.generations = generations  # 더 많은 세대로 최적해 탐색 강화
        self.mutation_rate = mutation_rate  # 높은 mutation rate로 지역 최적해 탈출 강화
        self.tournament_size = tournament_size  # 더 큰 tournament size로 선택 압력 증가
        self._center_positions = ((1,1), (1,2), (2,1), (2,2))
        self._corner_positions = ((0,0), (0,3), (3,0), (3,3))
        self._edge_positions = ((0,1), (0,2), (1,0), (1,3), (2,0), (2,3), (3,1), (3,2))
        self.start_time = 0
        self.max_time = 5  # 시간 제한 설정

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
        if self.is_new_memo:
            self.save_memoization()

    def check_timeout(self):
        if time.time() - self.start_time > self.max_time:
            raise TimeoutError("Computation time exceeded")

    def evaluate_fitness(self, piece):
        self.check_timeout()
        board_key = str(self.board.tolist())
        piece_key = str(piece)
        memo_key = f"{board_key}_{piece_key}"
        
        if memo_key in self.memoization:
            return self.memoization[memo_key]
            
        score = 0
        board_array = np.array(self.board)
        
        def check_characteristic_line(line):
            if 0 in line:
                pieces_in_line = [self.pieces[idx - 1] for idx in line if idx != 0]
                if not pieces_in_line:
                    return 0

                for char_idx in range(4):
                    chars = [p[char_idx] for p in pieces_in_line]
                    chars_set = set(chars)
                    if len(chars_set) == 1:
                        if len(chars) == 3:
                            return 5000  # 승리 직전 상태 보상 극대화
                        elif len(chars) == 2:
                            return 2000  # 잠재적 승리 가능성 보상 증가
                        else:
                            return 500  # 초기 승리 가능성 보상 증가
                return 100

            pieces_in_line = [self.pieces[idx - 1] for idx in line if idx != 0]
            for char_idx in range(4):
                if len(set(p[char_idx] for p in pieces_in_line)) == 1:
                    return 100000  # 승리 상태 보상 대폭 증가
            return 100

        # 가로, 세로 평가 - 가중치 증가
        for i in range(4):
            score += check_characteristic_line(board_array[i]) * 3.0
            score += check_characteristic_line(board_array[:, i]) * 3.0

        # 대각선 평가 - 더 높은 가중치
        score += check_characteristic_line([board_array[i][i] for i in range(4)]) * 4.0
        score += check_characteristic_line([board_array[i][3-i] for i in range(4)]) * 4.0

        # 2x2 서브그리드 평가 - 추가 패턴 탐지 강화
        for i in range(3):
            for j in range(3):
                subgrid = [
                    board_array[i][j], board_array[i][j+1],
                    board_array[i+1][j], board_array[i+1][j+1]
                ]
                score += check_characteristic_line(subgrid) * 3.0

        # 전략적 위치 점수 - 차등 가중치 강화
        piece_idx = self.pieces.index(piece) + 1
        
        # 중앙 위치 최우선
        score += sum(800 for pos in self._center_positions 
                    if board_array[pos[0]][pos[1]] == piece_idx)
        
        # 코너 위치 차선
        score += sum(400 for pos in self._corner_positions 
                    if board_array[pos[0]][pos[1]] == piece_idx)
        
        # 가장자리 위치 최하선
        score += sum(200 for pos in self._edge_positions 
                    if board_array[pos[0]][pos[1]] == piece_idx)

        # 방어 전략 강화
        for i in range(4):
            for j in range(4):
                if board_array[i][j] == 0:
                    board_array[i][j] = piece_idx
                    if self.check_win(board_array):
                        score -= 20000  # 상대방 승리 가능성 차단 강화
                    board_array[i][j] = 0

        # 연결성 평가 추가
        for i in range(4):
            for j in range(4):
                if board_array[i][j] != 0:
                    # 인접한 셀과의 특성 공유 평가
                    for di, dj in [(0,1), (1,0), (1,1), (1,-1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 4 and 0 <= nj < 4 and board_array[ni][nj] != 0:
                            piece1 = self.pieces[board_array[i][j] - 1]
                            piece2 = self.pieces[board_array[ni][nj] - 1]
                            shared_chars = sum(1 for k in range(4) if piece1[k] == piece2[k])
                            score += shared_chars * 300

        self.memoization[memo_key] = score
        return score

    def tournament_selection(self, population, scores):
        tournament_size = min(self.tournament_size, len(population))
        tournament = random.sample(list(zip(population, scores)), tournament_size)
        return max(tournament, key=lambda x: x[1])[0]

    def crossover(self, parent1, parent2):
        if not isinstance(parent1, list):
            parent1, parent2 = list(parent1), list(parent2)
        # 균등 교차로 변경하여 더 나은 유전자 조합 생성
        child = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return tuple(child)

    def mutate(self, piece):
        if random.random() < self.mutation_rate:
            piece = list(piece)
            # 적응적 돌연변이 - 성능에 따라 변이 강도 조절
            num_mutations = np.random.geometric(p=0.3)  # 기하분포로 변이 수 결정
            indices = random.sample(range(4), min(num_mutations, 4))
            for idx in indices:
                piece[idx] = 1 - piece[idx]
            return tuple(piece)
        return piece

    def optimize(self):
        self.start_time = time.time()
        try:
            population = random.sample(self.available_pieces,
                                     min(self.population_size, len(self.available_pieces)))
            
            best_overall_piece = None
            best_overall_score = float('-inf')

            # 엘리트 전략 강화
            elite_count = max(20, self.population_size // 4)
            stagnation_counter = 0
            prev_best_score = float('-inf')

            for generation in range(self.generations):
                self.check_timeout()
                scores = [self.evaluate_fitness(piece) for piece in population]
                
                # 현재 세대 최고 개체 확인 및 관리
                current_best_score = max(scores)
                if current_best_score > best_overall_score:
                    best_overall_score = current_best_score
                    best_overall_piece = population[scores.index(current_best_score)]
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # 적응적 진화 전략 - 정체 시 파라미터 동적 조정
                if stagnation_counter > 5:
                    self.mutation_rate = min(0.9, self.mutation_rate * 2.0)
                    self.tournament_size = max(10, self.tournament_size - 5)
                else:
                    self.mutation_rate = 0.2
                    self.tournament_size = 40

                new_population = []
                
                # 엘리트 보존 강화
                elite_indices = sorted(range(len(scores)), 
                                    key=lambda i: scores[i], 
                                    reverse=True)[:elite_count]
                new_population.extend([population[i] for i in elite_indices])

                # 새로운 세대 생성 - 다양성 유지
                while len(new_population) < self.population_size:
                    if random.random() < 0.85:  # 85% 교차
                        parent1 = self.tournament_selection(population, scores)
                        parent2 = self.tournament_selection(population, scores)
                        child = self.crossover(parent1, parent2)
                    else:  # 15% 무작위 생성
                        child = random.choice(self.available_pieces)
                    
                    child = self.mutate(child)
                    if child in self.available_pieces:
                        new_population.append(child)

                population = new_population

            return best_overall_piece, best_overall_score

        except TimeoutError:
            if best_overall_piece:
                return best_overall_piece, best_overall_score
            return random.choice(self.available_pieces), 0

    def select_piece(self):
        if len(self.available_pieces) <= 2:
            return self.available_pieces[0]
            
        best_piece, best_score = self.optimize()
        
        if best_score < 0:
            # 최적해가 없을 경우 확장된 탐욕적 선택
            candidates = random.sample(self.available_pieces, 
                                    min(15, len(self.available_pieces)))
            return max(candidates, key=self.evaluate_fitness)
            
        return best_piece

    def place_piece(self, selected_piece):
        self.start_time = time.time()
        try:
            available_locs = [(row, col) for row in range(4) for col in range(4) 
                             if self.board[row][col] == 0]
            
            best_pos = None
            best_score = float('-inf')
            piece_idx = self.pieces.index(selected_piece) + 1

            # 모든 가능한 위치에 대해 완전 탐색 및 평가
            for pos in available_locs:
                self.check_timeout()
                test_board = deepcopy(self.board)
                test_board[pos[0]][pos[1]] = piece_idx
                
                # 승리 가능한 수가 있다면 즉시 선택
                if self.check_win(test_board):
                    return pos
                    
                # 위치 평가 점수 계산 - 전략적 가중치 강화
                score = 0
                
                # 중앙 위치 최우선
                if pos in self._center_positions:
                    score += 800
                # 코너 위치 차선
                elif pos in self._corner_positions:
                    score += 400
                # 가장자리 위치 최하선
                elif pos in self._edge_positions:
                    score += 200
                    
                # 방어적 평가 강화
                for opp_piece in self.available_pieces:
                    if opp_piece != selected_piece:
                        opp_idx = self.pieces.index(opp_piece) + 1
                        test_board[pos[0]][pos[1]] = opp_idx
                        if self.check_win(test_board):
                            score += 8000  # 상대방 승리 저지 강화
                            
                # 연결성 평가
                for di, dj in [(0,1), (1,0), (1,1), (1,-1)]:
                    ni, nj = pos[0] + di, pos[1] + dj
                    if 0 <= ni < 4 and 0 <= nj < 4 and test_board[ni][nj] != 0:
                        piece1 = selected_piece
                        piece2 = self.pieces[test_board[ni][nj] - 1]
                        shared_chars = sum(1 for k in range(4) if piece1[k] == piece2[k])
                        score += shared_chars * 300
                            
                if score > best_score:
                    best_score = score
                    best_pos = pos
                    
            return best_pos if best_pos else random.choice(available_locs)

        except TimeoutError:
            return random.choice(available_locs)

    def check_win(self, board):
        def check_line(line):
            if 0 in line:
                return False
            characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])
            return any(len(set(characteristics[:, i])) == 1 for i in range(4))
            
        # 가로, 세로 확인
        for i in range(4):
            if check_line(board[i]) or check_line(board[:, i]):
                return True
                
        # 대각선 확인
        if check_line(np.diag(board)) or check_line(np.diag(np.fliplr(board))):
            return True
            
        # 2x2 서브그리드 확인
        for i in range(3):
            for j in range(3):
                subgrid = [
                    board[i][j], board[i][j+1],
                    board[i+1][j], board[i+1][j+1]
                ]
                if 0 not in subgrid:
                    characteristics = [self.pieces[idx - 1] for idx in subgrid]
                    if any(len(set(char[i] for char in characteristics)) == 1 for i in range(4)):
                        return True
        return False
