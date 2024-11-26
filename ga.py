import random
import numpy as np
import json
import os
import time

class GeneticAlgorithm:
    def __init__(self, board, available_pieces, current_piece=None, population_size=10, generations=15, mutation_rate=0.2, tournament_size=5):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.memo_file = "ga_memo.json"
        self.memoization = self.load_memoization()
        self.is_new_memo = not os.path.exists(self.memo_file)
        self.board = board
        self.available_pieces = available_pieces
        self.current_piece = current_piece
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

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

    def evaluate_fitness(self, piece):
        board_key = str(self.board.tolist())
        piece_key = str(piece)
        memo_key = f"{board_key}_{piece_key}"
        
        if memo_key in self.memoization:
            return self.memoization[memo_key]
            
        score = 0
        
        def check_characteristic_line(line):
            if 0 in line:
                pieces_in_line = [self.pieces[idx - 1] for idx in line if idx != 0]
                if not pieces_in_line:
                    return 0

                for char_idx in range(4):
                    chars = [p[char_idx] for p in pieces_in_line]
                    if len(set(chars)) == 1:
                        if len(chars) == 3:
                            return 100  # Increased reward for near-win
                        elif len(chars) == 2:
                            return 40   # Increased reward for potential win
                return 5

            pieces_in_line = [self.pieces[idx - 1] for idx in line if idx != 0]
            for char_idx in range(4):
                if len(set(p[char_idx] for p in pieces_in_line)) == 1:
                    return 2000  # Increased win reward
            return 10

        # Evaluate rows and columns
        for i in range(4):
            score += check_characteristic_line(self.board[i])
            score += check_characteristic_line(self.board[:, i])

        # Evaluate diagonals
        diag1 = [self.board[i][i] for i in range(4)]
        diag2 = [self.board[i][3-i] for i in range(4)]
        score += check_characteristic_line(diag1)
        score += check_characteristic_line(diag2)

        # Evaluate 2x2 subgrids
        for i in range(3):
            for j in range(3):
                subgrid = [
                    self.board[i][j], self.board[i][j+1],
                    self.board[i+1][j], self.board[i+1][j+1]
                ]
                score += check_characteristic_line(subgrid)

        # Strategic position scoring
        center_positions = [(1,1), (1,2), (2,1), (2,2)]
        corner_positions = [(0,0), (0,3), (3,0), (3,3)]
        
        for pos in center_positions:
            if self.board[pos[0]][pos[1]] == self.pieces.index(piece) + 1:
                score += 25  # Increased center position reward
                
        for pos in corner_positions:
            if self.board[pos[0]][pos[1]] == self.pieces.index(piece) + 1:
                score += 15  # Added corner position reward

        self.memoization[memo_key] = score
        return score

    def tournament_selection(self, population, scores):
        tournament = random.sample(list(zip(population, scores)), 
                                 min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: x[1])[0]

    def crossover(self, parent1, parent2):
        if not isinstance(parent1, list):
            parent1, parent2 = list(parent1), list(parent2)
        points = sorted(random.sample(range(len(parent1)), 2))
        child = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
        return tuple(child)

    def mutate(self, piece):
        if random.random() < self.mutation_rate:
            piece = list(piece)
            num_mutations = random.randint(1, 2)
            indices = random.sample(range(4), num_mutations)
            for idx in indices:
                piece[idx] = 1 - piece[idx]
            return tuple(piece)
        return piece

    def optimize(self):
        population = random.sample(self.available_pieces,
                                 min(self.population_size, len(self.available_pieces)))
        
        best_overall_piece = None
        best_overall_score = float('-inf')

        for generation in range(self.generations):
            scores = [self.evaluate_fitness(piece) for piece in population]
            
            for piece, score in zip(population, scores):
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_piece = piece

            new_population = []
            elite_count = max(2, self.population_size // 5)
            elite_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:elite_count]
            new_population.extend([population[i] for i in elite_indices])

            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, scores)
                parent2 = self.tournament_selection(population, scores)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                if child in self.available_pieces:
                    new_population.append(child)

            population = new_population

        return best_overall_piece, best_overall_score

    def select_piece(self):
        if len(self.available_pieces) <= 2:
            return self.available_pieces[0]
            
        best_piece, best_score = self.optimize()
        
        if best_score < 0:
            candidates = random.sample(self.available_pieces, 
                                    min(3, len(self.available_pieces)))
            return max(candidates, key=self.evaluate_fitness)
            
        return best_piece

    def place_piece(self):
        available_locs = [(row, col) for row in range(4) for col in range(4) 
                         if self.board[row][col] == 0]
        
        # Try to find winning move
        for pos in available_locs:
            test_board = self.board.copy()
            piece_idx = self.pieces.index(self.current_piece) + 1 if self.current_piece in self.pieces else None
            if piece_idx:
                test_board[pos[0]][pos[1]] = piece_idx
                if self.check_win(test_board):
                    return pos
        
        # Strategic positioning
        center_positions = [(1,1), (1,2), (2,1), (2,2)]
        corner_positions = [(0,0), (0,3), (3,0), (3,3)]
        
        for pos in center_positions:
            if pos in available_locs:
                return pos
                
        for pos in corner_positions:
            if pos in available_locs:
                return pos
                
        return random.choice(available_locs)

    def check_win(self, board):
        def check_line(line):
            if 0 in line:
                return False
            characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])
            return any(len(set(characteristics[:, i])) == 1 for i in range(4))
            
        # Check rows and columns
        for i in range(4):
            if check_line(board[i]) or check_line(board[:, i]):
                return True
                
        # Check diagonals
        if check_line(np.diag(board)) or check_line(np.diag(np.fliplr(board))):
            return True
            
        # Check 2x2 subgrids
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
