# 3D Gridworld Q-Learning

This project implements **tabular Q-learning** in a 3D Gridworld environment with stochastic transitions, absorbing terminals, and visualization of the learned value function and policy. It was developed as part of the 3D Gridworld Assignment evaluated under the course CS F407 Artificial Intelligence.

---

## 📌 Features
- **Environment (`Gridworld3D`)**
  - 3D grid with configurable size (default 6×6×6)
  - Obstacles randomly generated (10–15% of cells) with reproducibility
  - Goal (+50) and Pit (−50) absorbing states
  - Stochastic slip transitions: intended move with probability *p*, perpendicular slips with probability *(1-p)/4* each
  - Step cost (default −1)

- **Agent**
  - Tabular Q-learning with ε-greedy exploration
  - Linear epsilon decay schedule
  - Tracks rewards, epsilon values, and success rate during training

- **Evaluation**
  - Learned greedy policy vs. random baseline
  - Mean ± std returns over 100 evaluation episodes

- **Experiments**
  - Varying **discount factor γ**
  - Varying **slip probability p**
  - Varying **step cost**
  - Varying **learning rate α**
  - Plots of convergence curves and parameter sensitivity

- **Visualization**
  - 3D value function visualized as 2D heatmaps for z-slices
  - Policy arrows (±x, ±y) and markers for vertical moves (±z)
  - Special markers for start, goal, pit, and obstacles
  - Learning curves, epsilon decay, success rate

---

## 📂 Project Structure

```
.
├── main_notebook.ipynb # Full end-to-end notebook (Colab/Jupyter)
├── grid3d/ # Modular package
│ ├── init.py
│ ├── env.py # Gridworld3D environment
│ ├── qlearning.py # Q-learning algorithm
│ ├── eval.py # Evaluation functions
│ ├── experiments.py # Parameter variation experiments
│ └── viz.py # Visualization utilities
└── README.md # This file
```

---

## ⚙️ Installation

Create a Python environment (≥3.8 recommended):

```bash
conda create -n grid3d python=3.9 -y
conda activate grid3d
pip install numpy matplotlib
```

Or with pip only:
```
pip install numpy matplotlib
```

This directly installs dependencies into your current Python environment.

---

## ▶️ Usage

Run in **Jupyter / Colab**:

1. Open `main_notebook.ipynb`  
2. Run all cells to:
   - Initialize environment  
   - Train Q-learning agent  
   - Evaluate policy vs random baseline  
   - Run parameter sensitivity experiments  
   - Generate visualizations  

Run modular components in **Python**:

```python
from grid3d import Gridworld3D, train_q_learning, extract_greedy_policy
from grid3d import evaluate_policy, run_experiments, plot_value_slices_with_policy

env = Gridworld3D()
Q, info = train_q_learning(env, episodes=2000)
policy = extract_greedy_policy(Q)
mean, std = evaluate_policy(env, policy)
```

This allows component-wise usage without running the full notebook, enabling modular experiments and integration.

---

## 📊 Outputs
- **Training curves**: episode rewards, moving average, epsilon decay  
- **Success rate curve**: fraction of episodes reaching goal  
- **Evaluation metrics**: mean ± std return for learned vs random policy  
- **Parameter sensitivity**:
  - Discount factor γ  
  - Slip probability p  
  - Step cost  
  - Learning rate α  
- **3D visualization**:
  - Heatmaps of value function (max_a Q(s,a)) for multiple z-slices  
  - Policy arrows and markers  

---

## 📑 Deliverables
- Code: `grid3d/` package + `main_notebook.ipynb`  
- Experiments & Analysis: included in notebook  
- Visualizations: plots generated during notebook run  
- Report (PDF): to be written separately (≤4 pages)  
- `README.md`: this file  

---

## 🔁 Reproducibility
- Random seeds fixed (`SEED=42` for environment, `seed=123` for training)  
- Obstacle layout consistent across runs  
- Deterministic numpy RNG  

---

## 👥 Authors
- Darshbir Singh
- Avichal Dwivedi  
- Abheek Arora  

---

## 📌 Notes
- Default grid size is **6×6×6**, but can be adjusted.  
- Training for larger grids may require more episodes for convergence.  
