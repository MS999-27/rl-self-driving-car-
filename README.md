# rl-self-driving-car-
A reinforcement learning-based autonomous car navigation system using Soft Actor-Critic (SAC) algorithm. The car learns to navigate a dynamic environment, avoid obstacles, and reach target destinations through interaction.
# RL Self-Driving Car using SAC Algorithm

A reinforcement learning-based autonomous car navigation system using Soft Actor-Critic (SAC) algorithm. The car learns to navigate a dynamic environment, avoid obstacles, and reach target destinations through interaction.

## 🚗 Features

- **Soft Actor-Critic (SAC) Implementation** - State-of-the-art deep RL algorithm for continuous control
- **Real-time Visualization** - Kivy-based interactive GUI with obstacle painting
- **Sensor-based Navigation** - Three forward sensors detect obstacles in real-time
- **Dynamic Goal System** - Automatically relocates target when reached
- **Training Metrics** - Real-time reward tracking and entropy monitoring
- **Performance Reports** - Auto-generates training graphs (saved to Desktop)

## 🎮 How It Works

### Agent Architecture
- **Policy Network**: Neural network with 2 hidden layers (256 units each)
- **Action Space**: Continuous rotation (-1 to 1, scaled by ×15)
- **State Space**: 5 inputs
  - 3 forward sensor readings (left, center, right)
  - 2 orientation signals (forward/backward alignment with goal)

### Training Loop
1. Car reads sensor signals
2. Policy network outputs rotation action
3. Car moves and receives reward signal
4. Network updates based on SAC loss

### Reward System
- **+1.5**: Moving towards goal (distance < 400 pixels)
- **-0.5**: Moving away from goal
- **-100**: Collision or boundary violation (instant reset)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/rl-self-driving-car.git
cd rl-self-driving-car

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```bash
python Car_map_Game_File.py
```

### In-Game Controls
- **Left Mouse**: Draw obstacles on map (yellow lines)
- **SAVE REPORT Button**: Generate training performance graph

### UI Displays
- Current reward
- Penalty count (collisions/boundaries)
- Elapsed time

## 📊 Output

When you click "SAVE REPORT":
- Generates 2 plots:
  - **Training Rewards**: Shows reward progression
  - **Model Exploration**: Entropy history (exploration vs exploitation)
- Saved as `AI_Report_[timestamp].png` on Desktop

## 📁 File Descriptions

| File | Purpose |
|------|---------|
| `ai.py` | SAC algorithm implementation (PolicyNetwork + SAC class) |
| `Car_map_Game_File.py` | Main game loop, Kivy UI, and simulation engine |

## 🔧 Technical Details

### SAC Algorithm Components
- **Policy**: Outputs mean & log-std for Gaussian action distribution
- **Action Sampling**: Reparameterization trick with tanh squashing
- **Log Probability**: Corrected for tanh transformation
- **Entropy**: Tracked to monitor exploration

### Car Physics
- Velocity: 5.5 pixels/frame
- Size: 35×18 pixels
- Sensor range: ±10 pixel radius
- Rotation: Continuous from policy output

## 🎯 Potential Improvements

- [ ] Add Value/Critic networks for Q-learning
- [ ] Implement experience replay buffer
- [ ] Add lidar visualization
- [ ] Multi-goal navigation
- [ ] Save/load trained models
- [ ] Hyperparameter tuning interface

## 📝 License

This project is licensed under the MIT License - see LICENSE file.

## 👤 Author

**Mudita Songara**  
Electronics & Telecommunication Engineering (E&TC)  
Symbiosis Institute of Technology, Pune

## 📧 Contact

Feel free to open issues or reach out for collaborations!

---

**Status**: Active Development  
**Last Updated**: June 2026
