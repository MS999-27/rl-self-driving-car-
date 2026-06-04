# Architecture.md Content


# System Architecture: RL Self-Driving Car



```
┌─────────────────────────────────────────────────────────────┐
│                    KIVY GUI SIMULATION                       │
│  (Car_map_Game_File.py - Visualization & Game Loop)         │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────▼─────────┐
        │   Car Widget     │
        │  (Movement &     │
        │   Sensors)       │
        └────────┬─────────┘
                 │
        ┌────────▼────────────────┐
        │  Simulation.update()     │
        │  (State & Reward Loop)   │
        └────────┬────────────────┘
                 │
        ┌────────▼──────────┐
        │   SAC Agent       │
        │   (ai.py)         │
        │  PolicyNetwork    │
        └────────┬──────────┘
                 │
        ┌────────▼──────────┐
        │  Action Output    │
        │  (Rotation)       │
        └───────────────────┘
```

---

## 🤖 AI Component Architecture

### 1. PolicyNetwork (Neural Network)

```python
Input Layer (5 features)
    │
    ├─> Linear(5 → 256)
    ├─> ReLU Activation
    │
    ├─> Linear(256 → 256)
    ├─> ReLU Activation
    │
    ├─> Split into 2 outputs:
    │   ├─ mean:    Linear(256 → 1)
    │   └─ log_std: Linear(256 → 1)
    │
Output: (μ, σ) - Mean & Standard Deviation
```

**Parameters:**
- Hidden dimensions: 256 units
- Activation: ReLU
- Output constraint: log_std clamped [-20, 2]

### 2. SAC (Soft Actor-Critic) Agent

```python
SAC Class Structure:
├── PolicyNetwork
│   └── Generates continuous actions
│
├── Optimizer (Adam)
│   └── lr = 0.001
│
└── Entropy Tracking
    └── Monitors exploration
```

---

## 🎮 Game Loop & State Management

### State Pipeline

```
1. CAR SENSOR READINGS
   └─ 3 forward sensors (left, center, right)
      └─ Detects obstacle density in 20-pixel radius

2. ORIENTATION SIGNALS
   └─ 2 signals: forward/backward alignment with goal
      └─ Computed as angle between velocity & goal vector

3. STATE VECTOR (5D)
   ┌─────────────────────────────────┐
   │ [s1, s2, s3, orientation, -orientation] │
   └─────────────────────────────────┘
   s1, s2, s3 ∈ [0, 1] (sensor values)
   orientation ∈ [-1, 1] (angle/180)
```

### Action Output

```
Network Output: action ∈ [-1, 1]
                    │
                    └─> Scaled by 15
                    │
                    └─> Applied as rotation
                        (Δangle = action × 15)
```

### Reward Structure

```
COLLISION/BOUNDARY CHECK:
├─ If car hits obstacle or boundary
│  └─ Reward: -100 (instant penalty)
│  └─ Action: Reset to center with random rotation
│
DISTANCE CHECK:
├─ If distance to goal < 400 pixels
│  └─ Reward: +1.5 (moving towards goal)
│
├─ Else
│  └─ Reward: -0.5 (moving away)
│
GOAL REACHED:
└─ If distance < 70 pixels
   └─ Swap goal position
   └─ Continue episode
```

---

## 🚗 Car Physics & Sensing

### Car Representation

```
Visual:
┌─────────────┐
│   Car (35×18)│  Cyan Rectangle
└─────────────┘

Sensors (3 points):
        ↓ (center)
    ↙   │   ↖
  (left)    (right)
   -35°    +35°

Forward distance: 55 pixels
```

### Sensor Reading Algorithm

```python
For each sensor (3 total):
1. Calculate sensor position
   └─ Offset 55 pixels forward at angle ±35°

2. Query sand map
   └─ Check 20×20 pixel area around sensor
   └─ Sum all obstacle values

3. Normalize to [0, 1]
   └─ value / 400 (max pixel sum)

4. Boundary handling
   └─ If out of bounds → signal = 1.0
   └─ Else → signal = normalized_value
```

### Movement Physics

```
Velocity Vector:
├─ Magnitude: 5.5 pixels/frame
└─ Direction: car angle (updated each frame)

Position Update:
new_pos = old_pos + velocity

Rotation Update:
new_angle = old_angle + action_rotation
```

---

## 📊 Training Loop (60 FPS)

```
FRAME N:
│
├─ 1. Read Car Sensors
│      └─ Get 3 sensor values
│
├─ 2. Calculate Goal Alignment
│      └─ Compute orientation to goal
│
├─ 3. Update SAC Agent
│      └─ Input: state vector (5D)
│      └─ Output: rotation action, entropy
│
├─ 4. Move Car
│      └─ Apply rotation
│      └─ Update position
│
├─ 5. Collision Detection
│      └─ Check boundaries
│      └─ Check obstacle map
│
├─ 6. Calculate Reward
│      └─ Distance-based or collision penalty
│
├─ 7. Store Metrics
│      └─ Append reward to history
│      └─ Append entropy to history
│
└─ FRAME N+1
```

---

## 💾 Data Structures

### Global State Variables

```python
sand[w, h]              # 2D grid marking obstacles
                        # 0 = free space, 1 = obstacle

goal_x, goal_y          # Current target coordinates

scores[]                # Reward history (for plotting)

entropy_history[]       # Entropy values over time

last_reward             # Reward from last frame

penalty_count           # Total collisions/violations
```

### Car Widget Properties

```python
car.pos                 # Position (x, y)
car.angle               # Rotation angle (degrees)
car.velocity            # Velocity vector (vx, vy)
car.center              # Center point for rendering
car.size                # Dimensions (35, 18)
```

---

## 🎯 SAC Algorithm Details

### Action Sampling (Reparameterization Trick)

```
1. Forward pass through network
   └─ Get mean (μ) and log_std (σ_log)

2. Sample from Gaussian
   z ~ N(μ, σ)
   └─ z = μ + σ × ε  (ε ~ N(0,1))

3. Apply tanh squashing
   a = tanh(z)
   └─ Maps to [-1, 1]

4. Compute log probability (corrected for tanh)
   log_π = log_π_gaussian - Σ log(1 - a²)
```

### Entropy Calculation

```
Entropy = -E[log π(a|s)]

Purpose:
├─ High entropy: Explore (learn diverse behaviors)
├─ Low entropy: Exploit (use learned policy)
└─ Tracked to monitor learning progression
```

---

## 🖥️ Visualization System (Kivy)

### Widget Hierarchy

```
FinalApp (Main Application)
│
└─ root: FloatLayout
   │
   ├─ Painter Widget
   │  └─ Handles mouse drawing of obstacles
   │
   ├─ Simulation Widget
   │  └─ Car Widget (child)
   │     └─ Rendered at 60 FPS
   │
   └─ UI Label
      └─ Displays reward, penalties, time
```

### Rendering Pipeline

```
Canvas Background:
└─ Dark gray (0.1, 0.1, 0.15)

Sand/Obstacles:
└─ User-drawn yellow lines (width: 50px)
└─ Stored in sand[] array

Car:
├─ Body: Cyan rectangle
└─ Sensors: 3 colored circles
   ├─ Red (left)
   ├─ Green (center)
   └─ Blue (right)

Goal:
└─ Implicit (no visual, reset on reach)
```

---

## 📈 Metrics & Reporting

### Real-time Display (10 FPS)

```
┌─────────────────────┐
│ REWARD: +1.5        │
│ PENALTIES: 12       │
│ TIME: 05:32         │
└─────────────────────┘
```

### Report Generation (Matplotlib)

```
When "SAVE REPORT" clicked:

Graph 1: Training Rewards
├─ X-axis: Frame number
├─ Y-axis: Reward value
├─ Color: Cyan
└─ Shows learning curve

Graph 2: Model Exploration (Entropy)
├─ X-axis: Frame number
├─ Y-axis: Entropy value
├─ Color: Magenta
└─ Shows entropy decay over time

Output:
└─ PNG file saved to ~/Desktop/AI_Report_[timestamp].png
```

---

## ⚙️ Configuration Parameters

### Network Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `hidden_dim` | 256 | Network capacity |
| `learning_rate` | 0.001 | Adam optimizer step |
| `log_std_range` | [-20, 2] | Std clipping |
| `action_scale` | 15 | Rotation multiplier |

### Physics Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `velocity` | 5.5 px/frame | Car speed |
| `sensor_range` | 20 px | Detection radius |
| `sensor_distance` | 55 px | Forward offset |
| `sensor_angles` | [-35°, 0°, +35°] | Spread pattern |

### Reward Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `collision_penalty` | -100 | Crash penalty |
| `goal_reward` | +1.5 | Approach bonus |
| `penalty_reward` | -0.5 | Diverge penalty |
| `goal_threshold` | 70 px | Reach distance |

### Simulation Settings

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `FPS` | 60 | Game loop speed |
| `UI_update_rate` | 10 Hz | Label refresh |
| `window_width` | 1000 px | Screen width |
| `window_height` | 700 px | Screen height |

---

## 🔄 Training Dynamics

### Early Phase (0-5 min)
- Random policy → high entropy
- Car crashes frequently
- Rewards oscillate: mostly -100, some -0.5
- No consistent goal-seeking

### Middle Phase (5-20 min)
- Policy gradually improves
- Fewer collisions
- Entropy decreases
- More +1.5 rewards appear
- Car learns general navigation

### Late Phase (20+ min)
- Stable policy
- Low entropy (exploitation)
- Consistent positive rewards near goal
- Smooth navigation patterns
- May reach local optima

---

## 🚀 Potential Extensions

### Short-term
```
├─ Experience replay buffer
├─ Target network (stability)
├─ Critic network (Q-learning)
└─ Model checkpointing
```

### Medium-term
```
├─ Multiple agents
├─ Curriculum learning
├─ State normalization
└─ Recurrent policy (LSTM)
```

### Long-term
```
├─ Pixel-based observations (CNN)
├─ Multi-goal tasks
├─ Transfer learning
└─ Real-world deployment
```

---

## 📝 Notes

- **No experience replay**: Current SAC is on-policy only
- **Single critic limitation**: Only policy network, no value estimation
- **Sensor-based only**: No vision/image processing
- **Deterministic velocity**: Speed is fixed at 5.5 px/frame
- **2D movement only**: No vertical/3D motion

---

**Last Updated:** June 2026  
**Author:** Mudita Songara
```

