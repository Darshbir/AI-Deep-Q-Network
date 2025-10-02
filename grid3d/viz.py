# grid3d/viz.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_value_slices_with_policy(Q: np.ndarray, env, z_levels=None, figsize=(12, 4), cmap="viridis", savepath=None):
    if z_levels is None:
        # pick three evenly spaced slices
        z_levels = list(sorted(set([0, env.D//2, env.D-1])))
    V = Q.max(axis=1)
    # Build 3D value array with NaN for obstacles
    val = np.full((env.H, env.W, env.D), np.nan, dtype=float)
    for s, c in enumerate(env.idx2coord):
        val[c] = V[s]

    fig, axes = plt.subplots(1, len(z_levels), figsize=figsize, squeeze=False)
    axes = axes[0]

    # Precompute greedy actions per state
    greedy = Q.argmax(axis=1)

    for i, z in enumerate(z_levels):
        ax = axes[i]
        # Heatmap per slice z
        slice2d = val[:, :, z].T  # transpose to display (x along horizontal) nicely
        im = ax.imshow(slice2d, origin="lower", cmap=cmap)
        ax.set_title(f"z={z}  V= max_a Q(s,a)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Quiver arrows for in-plane actions (±x, ±y)
        Xs, Ys, U, Vv = [], [], [], []
        up_x, up_y, down_x, down_y = [], [], [], []
        for s, c in enumerate(env.idx2coord):
            x, y, zz = c
            if zz != z: 
                continue
            a = int(greedy[s])
            # in-plane arrows
            if a == 0:  # +x
                ux, vy = 1, 0
                Xs.append(x); Ys.append(y); U.append(ux); Vv.append(vy)
            elif a == 1:  # -x
                ux, vy = -1, 0
                Xs.append(x); Ys.append(y); U.append(ux); Vv.append(vy)
            elif a == 2:  # +y
                ux, vy = 0, 1
                Xs.append(x); Ys.append(y); U.append(ux); Vv.append(vy)
            elif a == 3:  # -y
                ux, vy = 0, -1
                Xs.append(x); Ys.append(y); U.append(ux); Vv.append(vy)
            elif a == 4:  # +z
                up_x.append(x); up_y.append(y)
            elif a == 5:  # -z
                down_x.append(x); down_y.append(y)

        if len(Xs) > 0:
            ax.quiver(np.array(Xs), np.array(Ys), np.array(U), np.array(Vv), color="white", angles='xy', scale_units='xy', scale=1)
        if len(up_x) > 0:
            ax.scatter(up_x, up_y, marker="^", c="lime", s=15, label="+z")
        if len(down_x) > 0:
            ax.scatter(down_x, down_y, marker="v", c="magenta", s=15, label="-z")
        # Mark terminals
        gx, gy, gz = env.goal
        if gz == z:
            ax.scatter([gx], [gy], marker="*", c="gold", s=120, label="goal")
        px, py, pz = env.pit
        if pz == z:
            ax.scatter([px], [py], marker="x", c="red", s=120, label="pit")
        # Obstacles
        for (ox, oy, oz) in env.obstacles:
            if oz == z:
                ax.scatter([ox], [oy], marker="s", c="black", s=10)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=160, bbox_inches="tight")
    return fig
