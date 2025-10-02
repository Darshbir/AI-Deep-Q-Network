# grid3d/env.py
from __future__ import annotations
import numpy as np
from collections import deque
from typing import Tuple, List, Dict, Optional, Set

Action = int
State = int
Coord = Tuple[int, int, int]

class Gridworld3D:
    """
    3D Gridworld MDP with slip:
    - States: all free (x,y,z) not in obstacles, plus absorbing goal/pit.
    - Actions: 6 moves: 0:+x,1:-x,2:+y,3:-y,4:+z,5:-z.
    - Transitions: intended axis move with prob p, 4 perpendicular moves with prob (1-p)/4 each; blocked moves stay in place.
    - Rewards: step cost c_step; entering goal gives +50 once, pit gives -50 once; terminals absorb afterwards with zero further reward.
    - Discount: gamma stored for reference.
    """
    ACTIONS = np.array([
        [1, 0, 0],   # +x
        [-1, 0, 0],  # -x
        [0, 1, 0],   # +y
        [0, -1, 0],  # -y
        [0, 0, 1],   # +z
        [0, 0, -1],  # -z
    ], dtype=int)

    def __init__(
        self,
        H: int = 6,
        W: int = 6,
        D: int = 6,
        p_intended: float = 0.85,
        c_step: float = -1.0,
        gamma: float = 0.95,
        start: Coord = (0, 0, 0),
        goal: Coord = (5, 5, 5),
        pit: Coord = (2, 2, 2),
        obstacle_ratio: float = 0.12,
        seed: int = 42,
        obstacles: Optional[Set[Coord]] = None,
    ):
        self.H, self.W, self.D = H, W, D
        self.shape = (H, W, D)
        self.n_actions = 6
        self.p_intended = float(p_intended)
        self.c_step = float(c_step)
        self.gamma = float(gamma)
        self.r_goal = 50.0
        self.r_pit = -50.0

        self.start = start
        self.goal = goal
        self.pit = pit
        self.rng = np.random.RandomState(seed)

        # Generate obstacles reproducibly and ensure connectivity from start to goal
        if obstacles is None:
            self.obstacles = self._generate_obstacles(obstacle_ratio)
        else:
            self.obstacles = set(obstacles)

        # Build state indexing for free cells (excluding obstacles)
        self.coord2idx: Dict[Coord, State] = {}
        self.idx2coord: List[Coord] = []
        self._build_indexing()

        assert self.start in self.coord2idx, "Start must be free"
        assert self.goal in self.coord2idx, "Goal must be free"
        assert self.pit in self.coord2idx, "Pit must be free"

        # Terminals (absorbing)
        self.terminals: Set[Coord] = {self.goal, self.pit}
        self._absorbing = False  # Flag used in step to emit terminal reward only once
        self.s: Coord = self.start

    def _neighbors6(self, c: Coord) -> List[Coord]:
        x, y, z = c
        neigh = []
        for dx, dy, dz in Gridworld3D.ACTIONS:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < self.H and 0 <= ny < self.W and 0 <= nz < self.D:
                neigh.append((nx, ny, nz))
        return neigh

    def _generate_obstacles(self, ratio: float) -> Set[Coord]:
        total = self.H * self.W * self.D
        n_obs = int(round(ratio * total))
        reserved = {self.start, self.goal, self.pit}
        all_cells = [(x, y, z) for x in range(self.H) for y in range(self.W) for z in range(self.D)]
        attempts = 0
        while True:
            attempts += 1
            self.rng.shuffle(all_cells)
            obs = set()
            for c in all_cells:
                if c in reserved: 
                    continue
                if len(obs) < n_obs:
                    obs.add(c)
            # Ensure connectivity from start to goal
            if self._has_path(self.start, self.goal, obs):
                # Ensure pit is reachable from start too (optional but useful)
                if self._has_path(self.start, self.pit, obs):
                    return obs
            if attempts > 200:
                # Relax obstacle count slightly if hard to satisfy
                n_obs = max(0, n_obs - 1)

    def _has_path(self, s: Coord, t: Coord, obs: Set[Coord]) -> bool:
        if s in obs or t in obs:
            return False
        dq = deque([s])
        seen = {s}
        while dq:
            u = dq.popleft()
            if u == t:
                return True
            for v in self._neighbors6(u):
                if v in obs or v in seen:
                    continue
                seen.add(v)
                dq.append(v)
        return False

    def _build_indexing(self):
        self.coord2idx.clear()
        self.idx2coord.clear()
        for x in range(self.H):
            for y in range(self.W):
                for z in range(self.D):
                    c = (x, y, z)
                    if c in self.obstacles:
                        continue
                    idx = len(self.idx2coord)
                    self.coord2idx[c] = idx
                    self.idx2coord.append(c)
        self.n_states = len(self.idx2coord)

    def reset(self, to: Optional[Coord] = None) -> State:
        self.s = self.start if to is None else to
        self._absorbing = (self.s in self.terminals)
        return self.coord2idx[self.s]

    def is_terminal(self, c: Coord) -> bool:
        return c in self.terminals

    def step(self, a: Action):
        """
        Returns: (s', r, done, info)
        s' as integer state id; 'done' True if terminal reached or currently in absorbing terminal.
        """
        if self._absorbing:
            # Already absorbing; self-loop with zero reward
            return self.coord2idx[self.s], 0.0, True, {}

        intended_vec = Gridworld3D.ACTIONS[a]
        axis = np.argmax(np.abs(intended_vec))
        # perpendicular moves: the other two axes have ± moves => 4 perpendicular actions
        perp_actions = []
        for ai, vec in enumerate(Gridworld3D.ACTIONS):
            if ai == a:
                continue
            # perpendicular iff different axis
            if np.argmax(np.abs(vec)) != axis:
                perp_actions.append(ai)
        assert len(perp_actions) == 4

        choices = [a] + perp_actions
        probs = [self.p_intended] + [ (1.0 - self.p_intended)/4.0 ] * 4

        # Sample action outcome
        pick = self.rng.choice(len(choices), p=probs)
        a_exec = choices[ int(pick) ]

        # Compute next coordinate with blocking
        x, y, z = self.s
        dx, dy, dz = Gridworld3D.ACTIONS[a_exec]
        nx, ny, nz = x + dx, y + dy, z + dz
        candidate = (nx, ny, nz)
        if not (0 <= nx < self.H and 0 <= ny < self.W and 0 <= nz < self.D):
            s_next = self.s
        elif candidate in self.obstacles:
            s_next = self.s
        else:
            s_next = candidate

        # Rewards and termination
        done = False
        r = self.c_step
        if s_next == self.goal:
            r = self.r_goal
            done = True
            self._absorbing = True
        elif s_next == self.pit:
            r = self.r_pit
            done = True
            self._absorbing = True

        self.s = s_next
        return self.coord2idx[self.s], float(r), bool(done), {}

    # Utilities
    def idx(self, c: Coord) -> State:
        return self.coord2idx[c]

    def coord(self, s: State) -> Coord:
        return self.idx2coord[s]

    def all_free_coords(self) -> List[Coord]:
        return list(self.idx2coord)

    def action_count(self) -> int:
        return self.n_actions
