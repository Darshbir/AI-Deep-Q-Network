# grid3d/experiments.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple
from .env import Gridworld3D
from .qlearning import train_q_learning, extract_greedy_policy
from .eval import evaluate_policy, evaluate_random

def run_experiments(
    base_env: Gridworld3D,
    gamma_list: List[float],
    p_list: List[float],
    step_costs: List[float],
    episodes: int = 4000,
    alpha: float = 0.1,
    epsilon_sched: Tuple[float,float,int] = (1.0, 0.05, 3000),
    max_steps_per_ep: int = 200,
    seed: int = 1234,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    results: Dict[str, Any] = {"gamma": {}, "p": {}, "step": {}}

    # Fix environment topology (obstacles) for all runs
    fixed_obstacles = set(base_env.obstacles)
    H, W, D = base_env.H, base_env.W, base_env.D
    start, goal, pit = base_env.start, base_env.goal, base_env.pit

    eps0, eps1, edecay = epsilon_sched

    # Vary gamma
    for g in gamma_list:
        env = Gridworld3D(H, W, D,
                          p_intended=base_env.p_intended,
                          c_step=base_env.c_step,
                          gamma=g,
                          start=start, goal=goal, pit=pit,
                          seed=seed, obstacles=fixed_obstacles)
        Q, info = train_q_learning(env, episodes=episodes, alpha=alpha, gamma=g,
                                   epsilon_start=eps0, epsilon_end=eps1, epsilon_decay_steps=edecay,
                                   max_steps_per_ep=max_steps_per_ep, seed=seed)
        pi = extract_greedy_policy(Q)
        mean_g, std_g = evaluate_policy(env, pi, episodes=100, max_steps_per_ep=max_steps_per_ep, seed=seed+1)
        rnd_mean, rnd_std = evaluate_random(env, episodes=100, max_steps_per_ep=max_steps_per_ep, seed=seed+2)
        results["gamma"][g] = {"mean": mean_g, "std": std_g, "random_mean": rnd_mean, "random_std": rnd_std, "rewards": info["rewards"]}

    # Vary p_intended
    for p in p_list:
        env = Gridworld3D(H, W, D,
                          p_intended=p,
                          c_step=base_env.c_step,
                          gamma=base_env.gamma,
                          start=start, goal=goal, pit=pit,
                          seed=seed, obstacles=fixed_obstacles)
        Q, info = train_q_learning(env, episodes=episodes, alpha=alpha, gamma=base_env.gamma,
                                   epsilon_start=eps0, epsilon_end=eps1, epsilon_decay_steps=edecay,
                                   max_steps_per_ep=max_steps_per_ep, seed=seed)
        pi = extract_greedy_policy(Q)
        mean_g, std_g = evaluate_policy(env, pi, episodes=100, max_steps_per_ep=max_steps_per_ep, seed=seed+1)
        rnd_mean, rnd_std = evaluate_random(env, episodes=100, max_steps_per_ep=max_steps_per_ep, seed=seed+2)
        results["p"][p] = {"mean": mean_g, "std": std_g, "random_mean": rnd_mean, "random_std": rnd_std, "rewards": info["rewards"]}

    # Vary step cost
    for cstep in step_costs:
        env = Gridworld3D(H, W, D,
                          p_intended=base_env.p_intended,
                          c_step=cstep,
                          gamma=base_env.gamma,
                          start=start, goal=goal, pit=pit,
                          seed=seed, obstacles=fixed_obstacles)
        Q, info = train_q_learning(env, episodes=episodes, alpha=alpha, gamma=base_env.gamma,
                                   epsilon_start=eps0, epsilon_end=eps1, epsilon_decay_steps=edecay,
                                   max_steps_per_ep=max_steps_per_ep, seed=seed)
        pi = extract_greedy_policy(Q)
        mean_g, std_g = evaluate_policy(env, pi, episodes=100, max_steps_per_ep=max_steps_per_ep, seed=seed+1)
        rnd_mean, rnd_std = evaluate_random(env, episodes=100, max_steps_per_ep=max_steps_per_ep, seed=seed+2)
        results["step"][cstep] = {"mean": mean_g, "std": std_g, "random_mean": rnd_mean, "random_std": rnd_std, "rewards": info["rewards"]}

    return results

def run_alpha_experiment(
    base_env,
    alpha_list=[0.05, 0.1, 0.2],
    episodes: int = 2000,
    gamma: float = 0.95,
    epsilon_sched=(1.0, 0.05, 3000),
    max_steps_per_ep: int = 200,
    seed: int = 123,
):
    results = {}
    eps0, eps1, edecay = epsilon_sched
    for a in alpha_list:
        Q, info = train_q_learning(base_env,
                                   episodes=episodes,
                                   alpha=a,
                                   gamma=gamma,
                                   epsilon_start=eps0,
                                   epsilon_end=eps1,
                                   epsilon_decay_steps=edecay,
                                   max_steps_per_ep=max_steps_per_ep,
                                   seed=seed)
        pi = extract_greedy_policy(Q)
        mean_g, std_g = evaluate_policy(base_env, pi, episodes=100, max_steps_per_ep=max_steps_per_ep, seed=seed+1)
        results[a] = {"mean": mean_g, "std": std_g, "rewards": info["rewards"]}
    return results
