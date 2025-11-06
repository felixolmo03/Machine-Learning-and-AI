import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
import os
from datetime import datetime

# Initialize Pygame and Constants

pygame.init()

# Colors (RGB)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)
RED = (255, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (50, 50, 50)

# Game dimensions

BLOCK_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
SCREEN_WIDTH = BLOCK_SIZE * (GRID_WIDTH + 8)
SCREEN_HEIGHT = BLOCK_SIZE * GRID_HEIGHT

# Tetromino shapes

SHAPES = [
    np.array([[1, 1, 1, 1]], dtype=int),  # I
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),  # T
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),  # L
    np.array([[1, 1, 1], [0, 0, 1]], dtype=int),  # J
    np.array([[1, 1], [1, 1]], dtype=int),  # O
    np.array([[0, 1, 1], [1, 1, 0]], dtype=int),  # S
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int)  # Z
]
SHAPE_COLORS = [CYAN, PURPLE, ORANGE, BLUE, YELLOW, GREEN, RED]
SHAPE_NAMES = ["I", "T", "L", "J", "O", "S", "Z"]

# Define actions

ACTION_MOVE_LEFT = 0
ACTION_MOVE_RIGHT = 1
ACTION_ROTATE = 2
ACTION_HARD_DROP = 3

ACTION_NAMES = {
    ACTION_MOVE_LEFT: "Left",
    ACTION_MOVE_RIGHT: "Right",
    ACTION_ROTATE: "Rotate",
    ACTION_HARD_DROP: "Hard Drop"
}

NUM_ACTIONS = 4

class DQNetwork(nn.Module):
    """Deep Q-Network for Tetris"""
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent for Tetris"""
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.update_target_every = 1000
        
        # Networks
        self.policy_net = DQNetwork(state_size, action_size).to(device)
        self.target_net = DQNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(capacity=10000)
        self.steps = 0
        
    def get_state_features(self, game):
        """Extract features from game state"""
        grid = game.grid.copy()
        
        # Column heights
        heights = []
        for col in range(GRID_WIDTH):
            height = 0
            for row in range(GRID_HEIGHT):
                if grid[row, col] != 0:
                    height = GRID_HEIGHT - row
                    break
            heights.append(height)
        
        # Holes (empty cells with filled cells above)
        holes = 0
        for col in range(GRID_WIDTH):
            block_found = False
            for row in range(GRID_HEIGHT):
                if grid[row, col] != 0:
                    block_found = True
                elif block_found and grid[row, col] == 0:
                    holes += 1
        
        # Bumpiness (height differences between adjacent columns)
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
        
        # Total height
        total_height = sum(heights)
        
        # Max height
        max_height = max(heights) if heights else 0
        
        # Complete lines
        complete_lines = sum(1 for row in grid if all(row))
        
        # Create feature vector
        features = heights + [holes, bumpiness, total_height, max_height, complete_lines]
        return np.array(features, dtype=np.float32)

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

    def save(self, filepath):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)

    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

class Tetris:
    def __init__(self):
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.current_piece = self.new_piece()
        self.next_piece = self.new_piece()
        self.game_over = False
        self.score = 0
        self.lines_cleared = 0
        self.fall_time = 0
        self.fall_speed = 500
        self.last_action = None
        self.pieces_placed = 0

    def new_piece(self):
        shape_idx = random.randint(0, len(SHAPES) - 1)
        return {
            'shape': SHAPES[shape_idx].copy(),
            'color': SHAPE_COLORS[shape_idx],
            'name': SHAPE_NAMES[shape_idx],
            'x': GRID_WIDTH // 2 - SHAPES[shape_idx].shape[1] // 2,
            'y': 0
        }

    def valid_move(self, piece, x_offset=0, y_offset=0, rotated_shape=None):
        shape_to_check = piece['shape'] if rotated_shape is None else rotated_shape
        for y, row in enumerate(shape_to_check):
            for x, cell in enumerate(row):
                if cell:
                    new_x = piece['x'] + x + x_offset
                    new_y = piece['y'] + y + y_offset
                    if (new_x < 0 or new_x >= GRID_WIDTH or
                            new_y >= GRID_HEIGHT or
                            (new_y >= 0 and self.grid[new_y, new_x])):
                        return False
        return True

    def rotate_piece(self):
        rotated_shape = np.rot90(self.current_piece['shape'], 3)
        if self.valid_move(self.current_piece, rotated_shape=rotated_shape):
            self.current_piece['shape'] = rotated_shape
            return True
        return False

    def move_piece(self, dx, dy):
        if self.valid_move(self.current_piece, x_offset=dx, y_offset=dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            return True
        return False

    def lock_piece(self):
        for y, row in enumerate(self.current_piece['shape']):
            for x, cell in enumerate(row):
                if cell:
                    grid_y = self.current_piece['y'] + y
                    grid_x = self.current_piece['x'] + x
                    if grid_y >= 0:
                        color_idx = SHAPE_COLORS.index(self.current_piece['color']) + 1
                        self.grid[grid_y, grid_x] = color_idx
        
        self.pieces_placed += 1
        lines = self.clear_lines()
        
        self.current_piece = self.next_piece
        self.next_piece = self.new_piece()
        
        if not self.valid_move(self.current_piece):
            self.game_over = True
        
        return lines

    def clear_lines(self):
        lines_cleared = 0
        y = GRID_HEIGHT - 1
        while y >= 0:
            if all(self.grid[y, :]):
                lines_cleared += 1
                for y2 in range(y, 0, -1):
                    self.grid[y2] = self.grid[y2 - 1]
                self.grid[0] = np.zeros(GRID_WIDTH, dtype=int)
            else:
                y -= 1

        self.lines_cleared += lines_cleared
        if lines_cleared == 1:
            self.score += 100
        elif lines_cleared == 2:
            self.score += 300
        elif lines_cleared == 3:
            self.score += 500
        elif lines_cleared == 4:
            self.score += 800

        return lines_cleared

    def update(self, delta_time):
        if self.game_over:
            return

        self.fall_time += delta_time
        if self.fall_time >= self.fall_speed:
            self.fall_time = 0
            if not self.move_piece(0, 1):
                self.lock_piece()

    def draw(self, surface):
        surface.fill(BLACK)
        
        # Draw grid background
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                pygame.draw.rect(surface, DARK_GRAY, 
                                (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(surface, GRAY, 
                                (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)
        
        # Draw placed blocks
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y, x]:
                    color_idx = self.grid[y, x] - 1
                    pygame.draw.rect(surface, SHAPE_COLORS[color_idx],
                                     (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(surface, WHITE,
                                     (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

        # Draw current piece
        if not self.game_over:
            for y, row in enumerate(self.current_piece['shape']):
                for x, cell in enumerate(row):
                    if cell:
                        pygame.draw.rect(surface, self.current_piece['color'],
                                         ((self.current_piece['x'] + x) * BLOCK_SIZE,
                                          (self.current_piece['y'] + y) * BLOCK_SIZE,
                                          BLOCK_SIZE, BLOCK_SIZE))
                        pygame.draw.rect(surface, WHITE,
                                         ((self.current_piece['x'] + x) * BLOCK_SIZE,
                                          (self.current_piece['y'] + y) * BLOCK_SIZE,
                                          BLOCK_SIZE, BLOCK_SIZE), 1)

        # Draw grid lines
        for x in range(GRID_WIDTH + 1):
            pygame.draw.line(surface, GRAY, (x * BLOCK_SIZE, 0), (x * BLOCK_SIZE, SCREEN_HEIGHT))
        for y in range(GRID_HEIGHT + 1):
            pygame.draw.line(surface, GRAY, (0, y * BLOCK_SIZE), (GRID_WIDTH * BLOCK_SIZE, y * BLOCK_SIZE))

        # Draw sidebar
        sidebar_x = GRID_WIDTH * BLOCK_SIZE + 5
        pygame.draw.line(surface, WHITE, (sidebar_x, 0), (sidebar_x, SCREEN_HEIGHT))

        # Draw next piece
        font = pygame.font.SysFont(None, 24)
        next_text = font.render("Next Piece:", True, WHITE)
        surface.blit(next_text, (sidebar_x + 10, 20))
        
        if not self.game_over:
            next_piece_x = sidebar_x + 30
            next_piece_y = 60
            for y, row in enumerate(self.next_piece['shape']):
                for x, cell in enumerate(row):
                    if cell:
                        pygame.draw.rect(surface, self.next_piece['color'],
                                         (next_piece_x + x * BLOCK_SIZE, 
                                          next_piece_y + y * BLOCK_SIZE,
                                          BLOCK_SIZE, BLOCK_SIZE))
                        pygame.draw.rect(surface, WHITE,
                                         (next_piece_x + x * BLOCK_SIZE, 
                                          next_piece_y + y * BLOCK_SIZE,
                                          BLOCK_SIZE, BLOCK_SIZE), 1)

        # Draw stats
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        surface.blit(score_text, (sidebar_x + 10, 120))
        
        lines_text = font.render(f"Lines: {self.lines_cleared}", True, WHITE)
        surface.blit(lines_text, (sidebar_x + 10, 150))
        
        pieces_text = font.render(f"Pieces: {self.pieces_placed}", True, WHITE)
        surface.blit(pieces_text, (sidebar_x + 10, 180))
        
        # Draw last action
        if self.last_action is not None:
            action_text = font.render(f"Action: {ACTION_NAMES.get(self.last_action, 'Unknown')}", True, YELLOW)
            surface.blit(action_text, (sidebar_x + 10, 220))

        if self.game_over:
            font = pygame.font.SysFont(None, 48)
            game_over_text = font.render("GAME OVER", True, RED)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 24))
            surface.blit(game_over_text, text_rect)

def train_headless(num_episodes=1000, save_every=100, model_path='tetris_dqn.pth'):
    """Train the DQN agent in headless mode"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Initialize agent
    state_size = 15  # 10 heights + holes + bumpiness + total_height + max_height + complete_lines
    agent = DQNAgent(state_size, NUM_ACTIONS, device)

    # Load existing model if available
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
            print(f"Loaded existing model from {model_path}")
        except:
            print("Could not load model, starting fresh")

    # Training stats
    scores = []
    lines_cleared_list = []
    pieces_placed_list = []

    print(f"\nStarting training for {num_episodes} episodes...")
    print("=" * 60)

    for episode in range(num_episodes):
        game = Tetris()
        state = agent.get_state_features(game)
        episode_reward = 0
        steps = 0
        
        while not game.game_over:
            # Select and perform action
            action = agent.select_action(state, training=True)
            game.last_action = action
            
            prev_score = game.score
            prev_lines = game.lines_cleared
            prev_height = max([sum(1 for row in range(GRID_HEIGHT) if game.grid[row, col] != 0) 
                             for col in range(GRID_WIDTH)])
            
            # Execute action
            if action == ACTION_MOVE_LEFT:
                game.move_piece(-1, 0)
            elif action == ACTION_MOVE_RIGHT:
                game.move_piece(1, 0)
            elif action == ACTION_ROTATE:
                game.rotate_piece()
            elif action == ACTION_HARD_DROP:
                while game.move_piece(0, 1):
                    pass
                lines = game.lock_piece()
            
            # Auto-drop every few steps
            if steps % 5 == 0:
                if not game.move_piece(0, 1):
                    if action != ACTION_HARD_DROP:
                        game.lock_piece()
            
            # Calculate reward
            reward = 0
            lines_cleared_now = game.lines_cleared - prev_lines
            if lines_cleared_now > 0:
                reward += lines_cleared_now ** 2 * 100  # Reward clearing lines heavily
            
            reward += (game.score - prev_score) * 0.1  # Small reward for score increase
            
            # Penalty for height
            curr_height = max([sum(1 for row in range(GRID_HEIGHT) if game.grid[row, col] != 0) 
                             for col in range(GRID_WIDTH)])
            if curr_height > prev_height:
                reward -= (curr_height - prev_height) * 2
            
            # Game over penalty
            if game.game_over:
                reward -= 100
            
            # Get next state
            next_state = agent.get_state_features(game)
            
            # Store transition
            agent.memory.push(state, action, reward, next_state, game.game_over)
            
            # Train
            loss = agent.train_step()
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        # Record stats
        scores.append(game.score)
        lines_cleared_list.append(game.lines_cleared)
        pieces_placed_list.append(game.pieces_placed)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            avg_lines = np.mean(lines_cleared_list[-10:])
            avg_pieces = np.mean(pieces_placed_list[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Score: {avg_score:.1f} | "
                  f"Avg Lines: {avg_lines:.1f} | "
                  f"Avg Pieces: {avg_pieces:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # Save model
        if (episode + 1) % save_every == 0:
            agent.save(model_path)
            print(f"Model saved to {model_path}")

    # Final save
    agent.save(model_path)
    print(f"\nTraining complete! Final model saved to {model_path}")
    print("=" * 60)
    print(f"Final stats (last 100 episodes):")
    print(f"  Average Score: {np.mean(scores[-100:]):.1f}")
    print(f"  Average Lines: {np.mean(lines_cleared_list[-100:]):.1f}")
    print(f"  Average Pieces: {np.mean(pieces_placed_list[-100:]):.1f}")

def play_with_trained_model(model_path='tetris_dqn.pth'):
    """Play Tetris with the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_size = 15
    agent = DQNAgent(state_size, NUM_ACTIONS, device)

    # Load model
    if os.path.exists(model_path):
        agent.load(model_path)
        agent.epsilon = 0  # No exploration during play
        print(f"Loaded model from {model_path}")
    else:
        print(f"No model found at {model_path}")
        return

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris DQN AI")
    clock = pygame.time.Clock()

    game = Tetris()
    running = True
    ai_active = True

    while running:
        delta_time = clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game = Tetris()
                elif event.key == pygame.K_a:
                    ai_active = not ai_active
        
        # AI control
        if ai_active and not game.game_over:
            if pygame.time.get_ticks() % 10 == 0:
                state = agent.get_state_features(game)
                action = agent.select_action(state, training=False)
                game.last_action = action
                
                if action == ACTION_MOVE_LEFT:
                    game.move_piece(-1, 0)
                elif action == ACTION_MOVE_RIGHT:
                    game.move_piece(1, 0)
                elif action == ACTION_ROTATE:
                    game.rotate_piece()
                elif action == ACTION_HARD_DROP:
                    while game.move_piece(0, 1):
                        pass
                    game.lock_piece()
        
        game.update(delta_time)
        game.draw(screen)
        
        # Display AI info
        font = pygame.font.SysFont(None, 24)
        ai_status = f"DQN AI: {'ACTIVE' if ai_active else 'PAUSED'} (Press A)"
        status_text = font.render(ai_status, True, GREEN if ai_active else RED)
        screen.blit(status_text, (10, SCREEN_HEIGHT - 60))
        
        small_font = pygame.font.SysFont(None, 18)
        controls = [
            "CONTROLS:",
            "A: Toggle AI",
            "R: Reset Game"
        ]
        for i, text in enumerate(controls):
            help_text = small_font.render(text, True, WHITE)
            screen.blit(help_text, (10, 10 + i * 20))
        
        pygame.display.flip()

    pygame.quit()

def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Training mode
        num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        train_headless(num_episodes=num_episodes)
    else:
        # Play mode
        play_with_trained_model()

if __name__ == "__main__":
    main()

