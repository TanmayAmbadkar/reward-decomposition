"""
Lunar Lander Reward Calibration Script
=======================================
Runs the built-in heuristic policy over many episodes to measure the
empirical distribution of each reward component and derived objective.

Use the printed percentile statistics to set normalisation constants
in mo_lunar_lander_utilities.py so that good episodes score near 1.0
on both O1 and O2.

Usage
-----
    python calibrate_lunar_lander.py
    python calibrate_lunar_lander.py --n_episodes 500 --continuous
"""

import argparse
import math
import numpy as np
from envs.lander import MOLunarLanderEnv
# ---------------------------------------------------------------------------
# Inline heuristic (copied from gymnasium source so we don't need gym.make)
# ---------------------------------------------------------------------------
def heuristic(env, s):
    angle_targ = s[0] * 0.5 + s[2] * 1.0
    angle_targ = np.clip(angle_targ, -0.4, 0.4)
    hover_targ = 0.55 * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5

    if s[6] or s[7]:
        angle_todo = 0
        hover_todo = -s[3] * 0.5

    if env.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


def run_calibration(n_episodes: int = 300, continuous: bool = False, seed_offset: int = 0):
    """
    Roll out the heuristic policy and collect per-episode statistics.

    Returns a dict of arrays, one value per episode:
        shaping_reward, fuel_cost, terminal_reward,
        O1_raw, O2_raw (before any scaling)
    """
    # Import here so the script fails loudly if box2d isn't installed
    try:
        from envs.lander import MOLunarLanderEnv
    except ImportError:
        raise ImportError(
            "mo_lunar_lander.py must be in the same directory. "
            "Make sure MOLunarLanderEnv is importable."
        )

    env = MOLunarLanderEnv(continuous=continuous)

    records = {
        "shaping":  [],
        "fuel":     [],   # always <= 0
        "terminal": [],
        "landed":   [],   # bool: terminal == +100
        "crashed":  [],   # bool: terminal == -100
        "timeout":  [],   # bool: terminal == 0
        "steps":    [],
    }

    print(f"Running heuristic policy for {n_episodes} episodes "
          f"(continuous={continuous}) ...")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + seed_offset)
        ep_shaping  = 0.0
        ep_fuel     = 0.0
        ep_terminal = 0.0
        ep_steps    = 0

        while True:
            action = heuristic(env, obs)
            obs, rvec, terminated, truncated, _ = env.step(action)

            ep_shaping  += float(rvec[0])
            ep_fuel     += float(rvec[1])
            ep_terminal += float(rvec[2])
            ep_steps    += 1

            if terminated or truncated:
                break

        records["shaping"].append(ep_shaping)
        records["fuel"].append(ep_fuel)
        records["terminal"].append(ep_terminal)
        records["landed"].append(ep_terminal >= 100.0)
        records["crashed"].append(ep_terminal <= -100.0)
        records["timeout"].append(abs(ep_terminal) < 100.0)
        records["steps"].append(ep_steps)

    env.close()
    return {k: np.array(v) for k, v in records.items()}


def print_statistics(records: dict):
    shaping  = records["shaping"]
    fuel     = records["fuel"]
    terminal = records["terminal"]
    fuel_abs = -fuel   # positive, higher = more fuel burned

    landed  = records["landed"]
    crashed = records["crashed"]
    timeout = records["timeout"]
    n       = len(shaping)

    print(f"\n{'='*60}")
    print(f"EPISODE OUTCOMES  (n={n})")
    print(f"{'='*60}")
    print(f"  Landed safely : {landed.sum():>4d}  ({100*landed.mean():.1f}%)")
    print(f"  Crashed       : {crashed.sum():>4d}  ({100*crashed.mean():.1f}%)")
    print(f"  Timeout       : {timeout.sum():>4d}  ({100*timeout.mean():.1f}%)")

    def stats(arr, name, subset=None, subset_name=""):
        if subset is not None:
            arr = arr[subset]
            tag = f" [{subset_name}]"
        else:
            tag = ""
        if len(arr) == 0:
            print(f"  {name}{tag}: no data")
            return
        print(
            f"  {name}{tag}:\n"
            f"    min={np.min(arr):>8.2f}  p10={np.percentile(arr,10):>8.2f}  "
            f"p25={np.percentile(arr,25):>8.2f}  median={np.median(arr):>8.2f}  "
            f"p75={np.percentile(arr,75):>8.2f}  p90={np.percentile(arr,90):>8.2f}  "
            f"max={np.max(arr):>8.2f}"
        )

    print(f"\n{'='*60}")
    print("RAW REWARD COMPONENTS (cumulative per episode)")
    print(f"{'='*60}")
    stats(shaping,  "shaping_reward ")
    stats(fuel_abs, "fuel_cost (abs)")
    stats(terminal, "terminal_reward")
    stats(records["steps"], "episode_steps  ")

    print(f"\n{'='*60}")
    print("RAW COMPONENTS — LANDED EPISODES ONLY")
    print(f"{'='*60}")
    stats(shaping,  "shaping_reward ", landed, "landed")
    stats(fuel_abs, "fuel_cost (abs)", landed, "landed")

    # O1 and O2 with CURRENT constants for comparison
    O1_CURRENT = (shaping + terminal) / 300.0
    O2_CURRENT = fuel_abs / 100.0

    print(f"\n{'='*60}")
    print("DERIVED OBJECTIVES — CURRENT SCALING (O1_SCALE=300, O2_SCALE=100)")
    print(f"{'='*60}")
    stats(O1_CURRENT, "O1 = (shaping+terminal)/300")
    stats(O2_CURRENT, "O2 = |fuel|/100            ")
    print(f"\n  Problem check — O2 for landed episodes:")
    stats(O2_CURRENT, "O2 (landed only)", landed, "landed")

    # Recommend better scaling
    # Goal: p90 of O2 for landed episodes ≈ 0.9 (good episodes score near 1)
    # Set O2_SCALE = p10 of fuel_abs for landed episodes (most efficient landings)
    landed_fuel = fuel_abs[landed]
    if len(landed_fuel) > 0:
        p10_fuel  = np.percentile(landed_fuel, 10)
        p50_fuel  = np.percentile(landed_fuel, 50)
        p90_fuel  = np.percentile(landed_fuel, 90)

        # Recommended O2_SCALE: scale so that the median landed episode
        # scores O2 = 0.7 (leaving headroom for unusually efficient episodes)
        recommended_o2_scale = p50_fuel / 0.7

        O2_RECOMMENDED = fuel_abs / recommended_o2_scale

        print(f"\n{'='*60}")
        print(f"RECOMMENDED SCALING")
        print(f"{'='*60}")
        print(f"  Landed fuel stats:")
        print(f"    p10 = {p10_fuel:.2f}  (most efficient landings)")
        print(f"    p50 = {p50_fuel:.2f}  (median landing)")
        print(f"    p90 = {p90_fuel:.2f}  (wasteful landings)")
        print(f"\n  Recommended O2_SCALE = {recommended_o2_scale:.1f}")
        print(f"  (sets median landed episode to O2 = 0.70)")
        print(f"\n  With O2_SCALE = {recommended_o2_scale:.1f}:")
        stats(O2_RECOMMENDED, "O2 recommended", landed, "landed")

        O1_landed = O1_CURRENT[landed]
        O2_rec_landed = O2_RECOMMENDED[landed]

        print(f"\n{'='*60}")
        print(f"UTILITY RANGES — LANDED EPISODES WITH RECOMMENDED SCALING")
        print(f"{'='*60}")

        # U1 FuelConstrainedLanding
        fuel_budget   = recommended_o2_scale * 0.5   # budget = O2 = 0.5
        quality_debt  = 0.2
        late_fee      = 0.05
        surplus       = O1_landed - quality_debt
        excess        = landed_fuel - fuel_budget
        late_pen      = np.where(excess > 0, excess**2 / recommended_o2_scale**2 + late_fee, 0)
        u1            = np.where(excess > 0, surplus - late_pen, surplus)
        stats(u1, "U1 FuelConstrained (landed)")

        # U2 JointSuccess
        u2 = O1_landed * O2_rec_landed
        stats(u2, "U2 JointSuccess   (landed)")

        # U3 SafetyFirst
        safety_threshold = 0.3
        safety_penalty   = -1.0
        u3 = np.where(O1_landed >= safety_threshold, O2_rec_landed, safety_penalty)
        stats(u3, "U3 SafetyFirst    (landed)")

        # Also show what crash episodes look like
        print(f"\n{'='*60}")
        print(f"UTILITY VALUES — CRASHED EPISODES")
        print(f"{'='*60}")
        if crashed.sum() > 0:
            O1_crashed = O1_CURRENT[crashed]
            fuel_crashed = fuel_abs[crashed]
            O2_crashed = fuel_crashed / recommended_o2_scale
            u2_crashed = O1_crashed * O2_crashed
            u3_crashed = np.where(O1_crashed >= safety_threshold, O2_crashed, safety_penalty)
            print(f"  U1: always = {-2.0} (crash_penalty)")
            stats(u2_crashed, "U2 JointSuccess   (crashed)")
            stats(u3_crashed, "U3 SafetyFirst    (crashed)")

        print(f"\n{'='*60}")
        print(f"COPY THESE VALUES INTO mo_lunar_lander_utilities.py")
        print(f"{'='*60}")
        print(f"  O1_SCALE = 300.0   # unchanged")
        print(f"  O2_SCALE = {recommended_o2_scale:.1f}")
        print(f"")
        print(f"  LLFuelConstrainedLanding:")
        print(f"    fuel_budget  = {fuel_budget:.1f}   # O2=0.5 threshold")
        print(f"    quality_debt = {quality_debt}")
        print(f"")
        print(f"  LLSafetyFirst:")
        print(f"    safety_threshold = {safety_threshold}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate Lunar Lander reward normalisation constants."
    )
    parser.add_argument(
        "--n_episodes", type=int, default=300,
        help="Number of heuristic episodes to run (default: 300)"
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Use continuous action variant"
    )
    parser.add_argument(
        "--seed_offset", type=int, default=0,
        help="Offset added to episode seeds for reproducibility"
    )
    args = parser.parse_args()

    records = run_calibration(
        n_episodes=args.n_episodes,
        continuous=args.continuous,
        seed_offset=args.seed_offset,
    )
    print_statistics(records)