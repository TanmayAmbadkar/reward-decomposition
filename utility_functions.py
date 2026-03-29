import numpy as np
import torch

class UtilityFunction:
    """Base class for utility functions."""
    def __call__(self, reward_vector):
        raise NotImplementedError

# ==========================================
# Deep Sea Treasure (DST) Utilities
# Input: [Treasure, Time_Penalty] (Time is negative)
# ==========================================

class DSTDebtUtility(UtilityFunction):
    """
    The specific 'Debt' scenario from MODeM Section 5.3.2.
    Scenario:
    - If treasure < debt: Prison (return e)
    - If treasure >= debt AND time <= deadline: Keep surplus (treasure - debt)
    - If treasure >= debt AND time > deadline: Pay late fees
    
    Default params from paper: d=20, t=19, e=-100, f=1
    """
    def __init__(self, debt=20.0, deadline=19.0, prison_penalty=-100.0, late_fee=1.0):
        self.d = debt
        self.t = deadline
        self.e = prison_penalty
        self.f = late_fee

    def __call__(self, r):
        if isinstance(r, np.ndarray):
            lib = np
        else:
            lib = torch

        treasure = r[..., 0]
        # In DST, time penalty is usually -1 per step. We need positive time steps.
        time_steps = lib.abs(r[..., 1]) 

        # 1. Paid debt on time
        cond1 = (treasure >= self.d) & (time_steps <= self.t)
        val1 = treasure - self.d

        # 2. Paid debt late
        cond2 = (treasure >= self.d) & (time_steps > self.t)
        penalty = (time_steps - self.t)**2 + self.f
        val2 = treasure - self.d - penalty

        # 3. Failed to pay debt (Prison)
        cond3 = (treasure < self.d)
        
        if lib == torch:
            val3 = torch.ones_like(treasure) * self.e
            return torch.where(cond1, val1, torch.where(cond2, val2, val3))
        else:
            return np.select([cond1, cond2, cond3], [val1, val2, self.e])

class DSTGeneralUtility(UtilityFunction):
    """
    Standard DST utilities for general benchmarking.
    Modes:
    0: 'linear': r0 + w * r1
    1: 'threshold': r0 if time > limit else penalty
    2: 'ratio': r0 / |r1| (Efficiency)
    """
    def __init__(self, mode='linear', linear_weight=0.5, time_limit=-19.0):
        self.mode = mode
        self.w = linear_weight
        self.limit = time_limit

    def __call__(self, r):
        if self.mode == 'linear':
            return r[..., 0] + self.w * r[..., 1]
            
        elif self.mode == 'threshold':
            # r[..., 1] is negative time. So r1 > -19 means time < 19 steps.
            if isinstance(r, np.ndarray):
                return np.where(r[..., 1] > self.limit, r[..., 0], -100.0)
            else:
                return torch.where(r[..., 1] > self.limit, r[..., 0], torch.tensor(-100.0, device=r.device))
                
        elif self.mode == 'ratio':
            # Treasure / Time
            time_abs = r[..., 1].abs() + 1e-6
            return r[..., 0] / time_abs
        else:
            raise ValueError(f"Unknown mode {self.mode}")

# ==========================================
# Fruit Tree Navigation (FTN) Utilities
# Input: [r0, r1, ...] (Nutrients)
# ==========================================

class FTNProductUtility(UtilityFunction):
    """
    The scalarization used for Single-Policy FTN experiments (MODeM 5.3.2).
    u(r) = Product(r_i)
    """
    def __call__(self, r):
        
        r0 = r[..., 0]/10
        r1 = r[..., 1]/10
        r2 = r[..., 2]/10
        return r0*r1*r2

class FTNMaxUtility(UtilityFunction):
    """
    Mode 1: Max
    Returns max(r0, r1) after normalizing inputs by dividing by 10.
    """
    def __call__(self, r):
        # Normalize inputs
        r0 = r[..., 0] / 10.0
        r1 = r[..., 1] / 10.0

        if isinstance(r, np.ndarray):
            return np.maximum(r0, r1)
        else:
            return torch.max(r0, r1)

class FTNMinUtility(UtilityFunction):
    """
    Mode 2: Min
    Returns min(r0, r1) after normalizing inputs by dividing by 10.
    """
    def __call__(self, r):
        r0 = r[..., 0] / 10.0
        r1 = r[..., 1] / 10.0

        if isinstance(r, np.ndarray):
            return np.minimum(r0, r1)
        else:
            return torch.min(r0, r1)

class FTN2ProductUtility(UtilityFunction):
    """
    Mode 3: Product
    Returns r0 * r1 after normalizing inputs by dividing by 10.
    """
    def __call__(self, r):
        r0 = r[..., 0] / 10.0
        r1 = r[..., 1] / 10.0
        return r0 * r1

class FTNMixedUtility(UtilityFunction):
    """
    Mode 4: Mixed
    Sorts the objectives and applies weights [2/3, 1/3] to the sorted values.
    Result is normalized by dividing by 10.
    """
    def __init__(self):
        self.weights_np = np.array([2/3, 1/3])
        self.weights_torch = torch.tensor([2/3, 1/3])

    def __call__(self, r):
        # We operate on the first two dimensions
        objs = r[..., :2]

        if isinstance(r, np.ndarray):
            # Sort along the last axis
            sorted_objs = np.sort(objs, axis=-1)
            # Dot product with weights
            return np.dot(sorted_objs, self.weights_np) / 10.0
        else:
            # Ensure weights are on the correct device
            if self.weights_torch.device != r.device:
                self.weights_torch = self.weights_torch.to(r.device)
            
            sorted_objs = torch.sort(objs, dim=-1).values
            
            if r.dim() == 1:
                return torch.dot(sorted_objs, self.weights_torch) / 10.0
            else:
                # Broadcasted sum/dot product
                return torch.sum(sorted_objs * self.weights_torch, dim=-1) / 10.0

class FTNDistanceUtility(UtilityFunction):
    """
    Mode 5: Distance
    Weighted negative squared distance to an ideal point.
    Uses raw objective values (unnormalized).
    """
    def __init__(self, w=(0.2, 0.8), ideal=(9.49729374, 8.77986293)):
        self.w = w
        self.i = ideal

    def __call__(self, r):
        # Use raw values (index 0 and 1)
        r0 = r[..., 0]
        r1 = r[..., 1]

        term0 = self.w[0] * (self.i[0] - r0)
        term1 = self.w[1] * (self.i[1] - r1)
        
        return -((term0 + term1) ** 2)

class NSWSpeedRatioUtility(UtilityFunction):
    """
    NSW Speed Ratio Utility.
    Robust implementation using log1p-like shift to avoid -inf and NaNs.
    Formula: log(r1 + 1) + log(r2 + 1) - log(-fuel + 1)
    """
    def __call__(self, vec):
        # We use a shift of 1.0 so u(0,0,0) = 0.
        # This avoids gradients exploding at 0 and keeps values well-scaled.
        c = 1e-10
        
        if isinstance(vec, np.ndarray):
            vec = torch.tensor(vec)
        
        if vec.dim() == 1:
            r0 = torch.clamp(vec[0], min=0)
            r1 = torch.clamp(vec[1], min=0)
            f = torch.clamp(vec[2], max=0) # Fuel is usually negative
            
            # log(r0 + 1) + log(r1 + 1) - log(-f + 1)
            return torch.log(r0 + c) + torch.log(r1 + c) - torch.log(-f + c) 
        else:
            vec_shape = vec.shape
            flat_vec = vec.view(-1,3)
            
            r0 = torch.clamp(flat_vec[:,0], min=0)
            r1 = torch.clamp(flat_vec[:,1], min=0)
            f = torch.clamp(flat_vec[:,2], max=0)

            resources = torch.log(r0 + c) + torch.log(r1 + c) 
            fuel = torch.log(-f + c)
            return (resources - fuel).view(vec_shape[:len(vec_shape) - 1])

# ==========================================
# Multi-Objective Lunar Lander Utilities
#
# Input reward vector (cumulative episode return):
#   r[..., 0] = shaping_reward  (trajectory quality, roughly -200 to +260)
#   r[..., 1] = fuel_cost       (always <= 0, cumulative engine cost)
#   r[..., 2] = terminal_reward (+100 safe landing, -100 crash, 0 timeout)
#
# Derived objectives:
#   O1 = (shaping + terminal) / 300
#        Landing quality. Clean landing ~ 0.93-1.10. Crash ~ -0.25 to -0.64.
#        Naturally separates crashes from landings via the terminal term.
#
#   O2 = clip(-fuel / 35.8, 0, 1.5)
#        Fuel efficiency. Scaled so median landed episode = 0.70.
#        Clipped at 1.5 to prevent anomalous high-fuel episodes dominating.
#        Efficient landing ~ 0.63-0.70. Wasteful landing ~ 0.82-0.89.
#
# Calibrated from 500 heuristic episodes (continuous=True):
#   Landed: 99.8%  |  shaping p50=212  |  fuel p50=25.07  |  steps p50=197
#
# Normalisation constants
# -----------------------
O1_SCALE = 300.0   # (max shaping ~260) + (terminal +100) gives O1 ~ 1.0-1.2
O2_SCALE = 35.8    # sets median landed episode to O2 = 0.70
O2_CLIP  = 1.5     # prevents anomalous episodes from dominating gradients
# ==========================================


def _compute_objectives(r):
    """
    Compute (O1, O2) from the raw cumulative reward vector.
    Handles numpy arrays and torch tensors, batched and unbatched.

    O1 = (shaping + terminal) / O1_SCALE
    O2 = clip(-fuel / O2_SCALE, 0, O2_CLIP)
    """
    O1 = (r[..., 0] + r[..., 2]) / O1_SCALE
    O2 = -r[..., 1] / O2_SCALE

    if isinstance(r, np.ndarray):
        O2 = np.clip(O2, 0.0, O2_CLIP)
    else:
        O2 = torch.clamp(O2, min=0.0, max=O2_CLIP)

    return O1, O2


# ==========================================
# U1 — Fuel-Constrained Landing
#
# "Land well, but you have a fuel budget."
#
# Mirrors DSTDebtUtility in structure — three branches:
#   crashed              → crash_penalty  (prison)
#   landed, fuel ok      → O1 - quality_debt  (surplus)
#   landed, over budget  → O1 - quality_debt - (excess² + late_fee)
#
# fuel_budget=25.0 is the p50 of landed fuel cost, splitting landed
# episodes roughly 50/50 between on-time and late-fee branches.
# This is the primary single-utility experiment function.
#
# Expected ranges (heuristic policy, continuous):
#   Good efficient landing : ~0.70-0.81
#   Good wasteful landing  : ~0.60-0.70 minus late penalty
#   Crash                  : -2.0 (hard)
# ==========================================

class LLFuelConstrainedLanding(UtilityFunction):
    """
    Fuel-constrained landing utility for Lunar Lander.

    Three-branch structure mirroring DSTDebtUtility:
      1. Crashed (terminal == -100):
             u = crash_penalty                           [hard failure]
      2. Landed safely, |fuel_cost| <= fuel_budget:
             u = O1 - quality_debt                      [surplus quality]
      3. Landed safely, |fuel_cost| > fuel_budget:
             u = O1 - quality_debt - (excess² + late_fee)  [penalised quality]

    Parameters
    ----------
    fuel_budget : float
        Maximum tolerable absolute fuel cost. Default 25.0 = p50 of landed
        heuristic episodes, creating a meaningful 50/50 on-time/late split.
    quality_debt : float
        Minimum acceptable O1. Episodes landing with poor trajectory quality
        are still penalised even without a fuel violation. Default 0.2.
    crash_penalty : float
        Utility for crash episodes. Default -2.0, well below the range of
        non-crash outcomes to create a hard disincentive.
    late_fee : float
        Additive constant in the over-budget fuel penalty. Default 0.05.
    """

    def __init__(
        self,
        fuel_budget: float   = 25.0,
        quality_debt: float  = 0.2,
        crash_penalty: float = -2.0,
        late_fee: float      = 0.05,
    ):
        self.fuel_budget   = fuel_budget
        self.quality_debt  = quality_debt
        self.crash_penalty = crash_penalty
        self.late_fee      = late_fee

    def __call__(self, r):
        O1, _    = _compute_objectives(r)
        terminal = r[..., 2]
        fuel_abs = -r[..., 1]                   # positive, more = worse

        # Excess is computed in raw fuel units, then normalised for the penalty
        excess   = fuel_abs - self.fuel_budget
        excess_n = excess / O2_SCALE            # normalise so penalty is ~O1-scale
        late_pen = excess_n ** 2 + self.late_fee
        surplus  = O1 - self.quality_debt

        if isinstance(r, np.ndarray):
            crashed     = terminal <= -100.0
            over_budget = (~crashed) & (excess > 0)
            return np.select(
                [crashed, over_budget],
                [self.crash_penalty, surplus - late_pen],
                default=surplus,
            )
        else:
            crashed     = terminal <= -100.0
            over_budget = (~crashed) & (excess > 0)
            crash_val   = torch.full_like(O1, self.crash_penalty)
            return torch.where(
                crashed,
                crash_val,
                torch.where(over_budget, surplus - late_pen, surplus),
            )


# ==========================================
# U2 — Joint Success
#
# "Land accurately AND use fuel efficiently, every single time."
#
# u = O1 * O2  with explicit crash guard
#
# The mathematically cleanest utility for demonstrating the SER/ESR
# covariance gap (Section 3.3):
#
#   JESR - JSER = Cov(O1, O2)
#
# A policy that crashes occasionally (O1 < 0) and is efficient otherwise
# produces a negative product on crash episodes, dragging ESR down.
# Under SER these average out. The product makes the covariance term
# directly measurable in per-episode (O1, O2) scatter plots.
#
# Crash episodes get explicit crash_penalty rather than the raw product
# (which could be small negative for some crashes) to ensure clean
# separation between crash and non-crash outcomes across both discrete
# and continuous variants.
#
# Expected ranges (heuristic policy, continuous):
#   Good efficient landing : ~0.59-0.72
#   Good wasteful landing  : ~0.82-0.99
#   Crash                  : -1.0 (explicit)
# ==========================================

class LLJointSuccess(UtilityFunction):
    """
    Product utility over landing quality and fuel efficiency.

    u(r) = O1 * O2              for non-crash episodes
           crash_penalty         for crash episodes (terminal == -100)

    where O1 = (shaping + terminal) / 300
          O2 = clip(-fuel / 35.8, 0, 1.5)

    Both objectives must be jointly high for the product to be large.
    Directly exposes Cov(O1, O2) — the quantity SER ignores and ESR rewards.

    Parameters
    ----------
    crash_penalty : float
        Explicit penalty for crash episodes. Default -1.0. Ensures clean
        separation from non-crash outcomes in both discrete and continuous
        variants where raw O1*O2 for crashes may be small negative.
    """

    def __init__(self, crash_penalty: float = -1.0):
        self.crash_penalty = crash_penalty

    def __call__(self, r):
        O1, O2   = _compute_objectives(r)
        terminal = r[..., 2]
        product  = O1 * O2

        if isinstance(r, np.ndarray):
            crashed = terminal <= -100.0
            return np.where(crashed, self.crash_penalty, product)
        else:
            crashed   = terminal <= -100.0
            crash_val = torch.full_like(product, self.crash_penalty)
            return torch.where(crashed, crash_val, product)


# ==========================================
# U3 — Safety First
#
# "I only care about fuel efficiency once safety is guaranteed."
#
# u = O2               if O1 >= safety_threshold
#     safety_penalty   otherwise
#
# Models a lexicographic-style preference where safety is a hard constraint
# and fuel efficiency is the secondary objective. The most distinctly
# safety-critical utility in the portfolio.
#
# Under SER: the policy can occasionally fall below the safety threshold
# as long as the average O1 is above it, keeping average fuel efficiency high.
# Under ESR: every single episode must clear the threshold or the hard
# penalty dominates, forcing consistent safety before fuel optimisation.
#
# safety_threshold=0.3 corresponds to (shaping + terminal) >= 90.
# All 35 crashes in the discrete calibration and the 1 in continuous
# fall below this — confirmed clean separation.
#
# Expected ranges (heuristic policy, continuous):
#   Safe episode   : O2 in [0.59, 0.89] → u in [0.59, 0.89]
#   Unsafe episode : -1.0 (hard)
# ==========================================

class LLSafetyFirst(UtilityFunction):
    """
    Safety-first utility for Lunar Lander.

    u(r) = O2                    if O1 >= safety_threshold
           safety_penalty         otherwise

    where O1 = (shaping + terminal) / 300
          O2 = clip(-fuel / 35.8, 0, 1.5)

    Safety is a hard constraint; fuel efficiency is the objective once
    safety is met. The starkest demonstration of the ESR/SER gap:
    SER can average over safety violations, ESR cannot tolerate any.

    Parameters
    ----------
    safety_threshold : float
        Minimum O1 for an episode to be considered safe. Default 0.3,
        confirmed to cleanly separate all crashes from all landings in
        calibration (crashes: O1 in [-0.64, -0.25], landings: O1 > 0.77).
    safety_penalty : float
        Utility for unsafe episodes. Default -1.0, well below the O2
        range of [0.59, 0.89] for safe episodes.
    """

    def __init__(
        self,
        safety_threshold: float = 0.3,
        safety_penalty: float   = -1.0,
    ):
        self.safety_threshold = safety_threshold
        self.safety_penalty   = safety_penalty

    def __call__(self, r):
        O1, O2 = _compute_objectives(r)

        if isinstance(r, np.ndarray):
            unsafe  = O1 < self.safety_threshold
            return np.where(unsafe, self.safety_penalty, O2)
        else:
            unsafe    = O1 < self.safety_threshold
            pen_val   = torch.full_like(O2, self.safety_penalty)
            return torch.where(unsafe, pen_val, O2)


# ==========================================
# Portfolio and single-utility exports
# ==========================================

# Multi-utility portfolio for multi-task / Table 1 style experiments
LLANDER_PORTFOLIO = [
    LLFuelConstrainedLanding(),
    LLJointSuccess(),
    LLSafetyFirst(),
]

# Primary single-utility for learning curve experiments (Fig 1 equivalent)
LLANDER_SINGLE_UTILITY = LLFuelConstrainedLanding()


# ==========================================
# Smoke test — verifies expected value ranges
# ==========================================
if __name__ == "__main__":
    # Representative episodes based on calibration statistics
    episodes = np.array([
        [ 212.0,  -25.0,  100.0],   # good efficient landing  (median)
        [ 212.0,  -32.0,  100.0],   # good wasteful landing   (p90 fuel)
        [  50.0,  -20.0, -100.0],   # crash
        [ 160.0,  -25.0,  100.0],   # marginal landing
    ], dtype=np.float32)

    labels = [
        "efficient landing",
        "wasteful landing ",
        "crash            ",
        "marginal landing ",
    ]

    utilities = [
        LLFuelConstrainedLanding(),
        LLJointSuccess(),
        LLSafetyFirst(),
    ]
    names = ["U1_FuelConstrained", "U2_JointSuccess   ", "U3_SafetyFirst    "]

    # Print O1, O2 for each episode
    print(f"\n{'Episode':<22} {'O1':>7} {'O2':>7}")
    print("-" * 38)
    for i, label in enumerate(labels):
        r = episodes[i]
        O1 = (r[0] + r[2]) / O1_SCALE
        O2 = float(np.clip(-r[1] / O2_SCALE, 0.0, O2_CLIP))
        print(f"{label:<22} {O1:>7.3f} {O2:>7.3f}")

    print(f"\n{'Episode':<22} " + " ".join(f"{n:>20}" for n in names))
    print("-" * (22 + 21 * len(utilities)))
    for i, label in enumerate(labels):
        r = episodes[i]
        vals = []
        for u in utilities:
            v = u(r)
            vals.append(f"{float(v):>20.4f}")
        print(f"{label:<22} " + " ".join(vals))

    print("\nBatched numpy:")
    for u, name in zip(utilities, names):
        result = u(episodes)
        arr = result if isinstance(result, np.ndarray) else result.numpy()
        print(f"  {name}: {arr}")

    print("\nBatched torch:")
    t_ep = torch.tensor(episodes)
    for u, name in zip(utilities, names):
        result = u(t_ep)
        print(f"  {name}: {result}")

    print("\nExpected approximate values:")
    print("  efficient landing: U1~0.72  U2~0.67  U3~0.70")
    print("  wasteful landing:  U1~0.60  U2~0.74  U3~0.74  (U1 pays late fee)")
    print("  crash:             U1=-2.0  U2=-1.0  U3=-1.0")
    print("  marginal landing:  U1~0.56  U2~0.55  U3~0.70")