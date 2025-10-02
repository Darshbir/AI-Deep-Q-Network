# grid3d/eval.py
from __future__ import annotations
import numpy as np
from typing import Callable, Tuple

def evaluate_policy(env, policy: np.ndarray, episodes: int = 100, max_steps_per_ep: int = 200, seed: int = 999) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    returns = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        G = 0.0
        steps = 0
        while not done and steps < max_steps_per_ep:
            a = int(policy[s])
            s, r, done, _ = env.step(a)
            G += r
            steps += 1
        returns.append(G)
    returns = np.array(returns, dtype=np.float32)
    return float(returns.mean()), float(returns.std(ddof=1))

def random_policy(nA: int):
    def pi(env, s, rng):
        return rng.randint(nA)
    return pi

def evaluate_random(env, episodes: int = 100, max_steps_per_ep: int = 200, seed: int = 2024) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    returns = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        G = 0.0
        steps = 0
        while not done and steps < max_steps_per_ep:
            a = rng.randint(env.n_actions)
            s, r, done, _ = env.step(a)
            G += r
            steps += 1
        returns.append(G)
    returns = np.array(returns, dtype=np.float32)
    return float(returns.mean()), float(returns.std(ddof=1))
