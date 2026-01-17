import pygame
import requests
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections

# --- CONFIGURATION ---
JAVA_URL = "http://localhost:8080/api/traffic"
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
BG_COLOR = (30, 30, 30)
ROAD_COLOR = (50, 50, 50)
CAR_COLOR = (255, 50, 50)     # Red for waiting cars
MOVING_COLOR = (50, 255, 50)  # Green for moving cars
GREEN_LIGHT = (0, 255, 0)
RED_LIGHT = (255, 0, 0)

# --- DQN PARAMS ---
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 2000
BATCH_SIZE = 32

# --- NEURAL NETWORK (Updated for 5 Inputs) ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- HELPER FUNCTIONS ---
def get_state_vector(axis, n, s, e, w):
    # Vector: [is_vertical, n_cars, s_cars, e_cars, w_cars]
    axis_val = 1.0 if axis == "VERTICAL" else 0.0
    return torch.FloatTensor([axis_val, n, s, e, w])

# --- INITIALIZATION ---
# Input dim is now 5 (Axis + 4 queues)
dqn = DQN(input_dim=5, output_dim=2) 
optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()
replay_buffer = collections.deque(maxlen=MEMORY_SIZE)
epsilon = EPSILON_START

# --- 4-WAY DRAWING FUNCTIONS ---
def draw_intersection(screen):
    # Vertical Road
    pygame.draw.rect(screen, ROAD_COLOR, (350, 0, 100, 800))
    # Horizontal Road
    pygame.draw.rect(screen, ROAD_COLOR, (0, 350, 800, 100))
    
    # Lane Dividers
    pygame.draw.line(screen, (255,255,255), (400, 0), (400, 350), 2) # North
    pygame.draw.line(screen, (255,255,255), (400, 450), (400, 800), 2) # South
    pygame.draw.line(screen, (255,255,255), (0, 400), (350, 400), 2) # West
    pygame.draw.line(screen, (255,255,255), (450, 400), (800, 400), 2) # East

def draw_cars(screen, n, s, e, w, axis):
    # North Queue (Top -> Center)
    for i in range(n):
        y = 330 - (i * 25)
        if y > 0: pygame.draw.circle(screen, CAR_COLOR, (375, y), 10)

    # South Queue (Bottom -> Center)
    for i in range(s):
        y = 470 + (i * 25)
        if y < 800: pygame.draw.circle(screen, CAR_COLOR, (425, y), 10)

    # West Queue (Left -> Center)
    for i in range(w):
        x = 330 - (i * 25)
        if x > 0: pygame.draw.circle(screen, CAR_COLOR, (x, 425), 10)

    # East Queue (Right -> Center)
    for i in range(e):
        x = 470 + (i * 25)
        if x < 800: pygame.draw.circle(screen, CAR_COLOR, (x, 375), 10)

def draw_lights(screen, axis):
    # Vertical Lights (North/South)
    color = GREEN_LIGHT if axis == "VERTICAL" else RED_LIGHT
    pygame.draw.circle(screen, color, (320, 320), 15) # North Light
    pygame.draw.circle(screen, color, (480, 480), 15) # South Light

    # Horizontal Lights (East/West)
    color = GREEN_LIGHT if axis == "HORIZONTAL" else RED_LIGHT
    pygame.draw.circle(screen, color, (320, 480), 15) # West Light
    pygame.draw.circle(screen, color, (480, 320), 15) # East Light

# --- MAIN LOOP ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("4-Way AI Traffic Control (Deep Q-Learning)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

running = True
last_state_vec = None
last_action = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    # 1. Get State
    try:
        data = requests.get(f"{JAVA_URL}/state", timeout=0.5).json()
    except: continue 

    n = data['north_cars']
    s = data['south_cars']
    e = data['east_cars']
    w = data['west_cars']
    axis = data['current_axis']
    
    current_state_vec = get_state_vector(axis, n, s, e, w)

    # 2. Train AI
    if last_state_vec is not None:
        reward = -1.0 * (n + s + e + w) # Minimize TOTAL traffic
        reward = reward / 20.0 # Normalize
        replay_buffer.append((last_state_vec, last_action, reward, current_state_vec))

    if len(replay_buffer) > BATCH_SIZE:
        batch = random.sample(replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        
        # Tensor conversion
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.FloatTensor(rewards)
        actions = torch.LongTensor(actions)
        
        q_values = dqn(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = dqn(next_states).max(1)[0]
        target_q = rewards + GAMMA * next_q
        
        loss = criterion(current_q, target_q.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epsilon > EPSILON_END: epsilon *= EPSILON_DECAY

    # 3. Choose Action
    if random.random() < epsilon:
        action = random.choice([0, 1])
    else:
        with torch.no_grad():
            action = dqn(current_state_vec).argmax().item()

    # 4. Act
    try:
        requests.post(f"{JAVA_URL}/action", json={"action": int(action)}, timeout=0.5)
    except: pass
    
    last_state_vec = current_state_vec
    last_action = action

    # 5. Visualize
    screen.fill(BG_COLOR)
    draw_intersection(screen)
    draw_cars(screen, n, s, e, w, axis)
    draw_lights(screen, axis)
    
    # Stats
    info = f"Eps: {epsilon:.2f} | Axis: {axis} | Total Cars: {n+s+e+w}"
    screen.blit(font.render(info, True, (255, 255, 255)), (10, 10))

    pygame.display.flip()
    clock.tick(10)

pygame.quit()