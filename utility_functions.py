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
# KEY DESIGN DECISION
# -------------------
# Utilities are formulated around shaping_reward and fuel_cost ONLY.
# terminal_reward is used exclusively as a crash detector (binary flag).
#
# Reason: terminal_reward = +100 only appears on successful landings.
# During early training the agent rarely lands, so utilities that require
# terminal = +100 to score positively provide no learning signal until
# the agent can already land — a chicken-and-egg problem.
#
# By using shaping as the primary quality signal, the utilities provide
# meaningful gradient information from the very first episode:
#   - Crashed episodes:  shaping typically negative, crash flag fires
#   - Timeout episodes:  shaping near zero or slightly positive
#   - Good flight:       shaping strongly positive (150-260)
#
# The SER/ESR gap argument is fully preserved:
#   SER can average crash episodes (large negative shaping) against
#   good flight episodes and look acceptable overall.
#   ESR penalises every crash episode individually, forcing the policy
#   to achieve consistent non-crash behavior.
#
# Calibrated from 500 heuristic episodes (continuous=True):
#   shaping p10=180  p50=212  p90=236  (landed episodes)
#   fuel    p10=22   p50=25   p90=32   (landed episodes)
#   crash rate: 0.2%  (heuristic)
#
# Normalisation constants
# -----------------------
SHAPING_SCALE = 200.0   # sets good flight episode to ~1.0
FUEL_SCALE    = 35.8    # sets median landed episode fuel efficiency to 0.70
FUEL_CLIP     = 1.5     # prevents anomalous high-fuel episodes dominating
# ==========================================
 
 
def _crashed(r):
    """Returns boolean mask: True where terminal_reward indicates a crash."""
    terminal = r[..., 2]
    if isinstance(r, np.ndarray):
        return terminal <= -100.0
    else:
        return terminal <= -100.0
 
 
def _shaping_norm(r):
    """Normalised shaping reward. Good flight ~ 1.0, crash ~ -0.5 to -1.0."""
    return r[..., 0] / SHAPING_SCALE
 
 
def _fuel_eff(r):
    """
    Normalised fuel efficiency. Higher = less fuel burned.
    Clipped to [0, FUEL_CLIP] so anomalous episodes don't dominate.
    """
    raw = -r[..., 1] / FUEL_SCALE
    if isinstance(r, np.ndarray):
        return np.clip(raw, 0.0, FUEL_CLIP)
    else:
        return torch.clamp(raw, min=0.0, max=FUEL_CLIP)
 
 
# ==========================================
# U1 — Crash-Penalised Trajectory Quality
#
# "Fly well. Don't crash."
#
# u = crash_penalty          if crashed
#     shaping / 200          otherwise
#
# The simplest utility that provides signal from episode 1:
#   - Any non-crash episode gets credit proportional to shaping quality
#   - Crash episodes get a hard penalty well below any non-crash score
#
# SER/ESR gap: SER can average crash episodes (-1.0) against good flight
# episodes (+1.0) and appear acceptable. ESR penalises every crash,
# forcing the policy to avoid crashes consistently.
#
# This is the primary single-utility experiment function (Fig 1 equivalent).
#
# Expected ranges:
#   Good flight (shaping~212)  :  ~1.06
#   Poor flight (shaping~0)    :  ~0.0
#   Crash                      :  -2.0
# ==========================================
 
class LLTrajectoryQuality(UtilityFunction):
    """
    Crash-penalised trajectory quality utility.
 
    u(r) = crash_penalty          if terminal == -100
           shaping / SHAPING_SCALE  otherwise
 
    Parameters
    ----------
    crash_penalty : float
        Utility for crash episodes. Default -2.0, well below the range
        of non-crash outcomes [-0.5, 1.3] to create a hard disincentive.
    """
 
    def __init__(self, crash_penalty: float = -2.0):
        self.crash_penalty = crash_penalty
 
    def __call__(self, r):
        shaping = _shaping_norm(r)
        crashed = _crashed(r)
 
        if isinstance(r, np.ndarray):
            return np.where(crashed, self.crash_penalty, shaping)
        else:
            crash_val = torch.full_like(shaping, self.crash_penalty)
            return torch.where(crashed, crash_val, shaping)
 
 
# ==========================================
# U2 — Crash-Penalised Joint Success
#
# "Fly well AND use fuel efficiently. Don't crash."
#
# u = crash_penalty              if crashed
#     shaping_norm * fuel_eff    otherwise
#
# The product utility makes the SER/ESR covariance gap explicit
# (Section 3.3 of the paper):
#
#   JESR - JSER = Cov(shaping_norm, fuel_eff)
#
# A SER policy can mix "good shaping, wasteful fuel" with "poor shaping,
# efficient fuel" episodes and average to an acceptable score.
# An ESR policy must achieve both jointly within every episode.
#
# Crash episodes return a hard penalty rather than the raw product
# (which could be negative-times-positive = negative, but inconsistently
# scaled) to ensure clean separation from non-crash outcomes.
#
# Expected ranges:
#   Good efficient flight  :  ~1.06 * 0.70 = ~0.74
#   Good wasteful flight   :  ~1.06 * 0.89 = ~0.94
#   Poor flight            :  ~0.0
#   Crash                  :  -1.0
# ==========================================
 
class LLJointSuccess(UtilityFunction):
    """
    Crash-penalised product utility over trajectory quality and fuel efficiency.
 
    u(r) = crash_penalty                        if terminal == -100
           shaping_norm * fuel_eff              otherwise
 
    where shaping_norm = shaping / 200
          fuel_eff      = clip(-fuel / 35.8, 0, 1.5)
 
    Directly exposes Cov(shaping_norm, fuel_eff) — the quantity SER ignores.
 
    Parameters
    ----------
    crash_penalty : float
        Explicit penalty for crash episodes. Default -1.0.
    """
 
    def __init__(self, crash_penalty: float = -1.0):
        self.crash_penalty = crash_penalty
 
    def __call__(self, r):
        shaping  = _shaping_norm(r)
        fuel     = _fuel_eff(r)
        crashed  = _crashed(r)
        product  = shaping * fuel
 
        if isinstance(r, np.ndarray):
            # return np.where(crashed, self.crash_penalty, product)
            return product
        else:
            crash_val = torch.full_like(product, self.crash_penalty)
            # return torch.where(crashed, crash_val, product)
            return product
 
 
# ==========================================
# U3 — Efficiency Under Safety
#
# "I only care about fuel efficiency once you're flying safely."
#
# u = crash_penalty        if crashed
#     poor_flight_penalty  if shaping < safety_threshold (flew poorly)
#     fuel_eff             otherwise (flew well — measure efficiency)
#
# Three meaningful levels that provide signal throughout training:
#   Level 1 (early training):  crash vs no-crash distinction
#   Level 2 (mid training):    poor vs good flight distinction
#   Level 3 (late training):   fuel efficiency optimisation
#
# This is the most distinctly safety-critical utility. Under SER the
# policy can occasionally fly poorly as long as average shaping is above
# the threshold. Under ESR every episode must clear the safety threshold
# before fuel efficiency matters.
#
# safety_threshold=0.0 means any episode with positive net shaping is
# considered "safe" — a low bar that the agent can clear early in training,
# giving U3 a meaningful non-crash signal from the start.
#
# Expected ranges:
#   Good efficient flight  :  fuel_eff ~ 0.70
#   Good wasteful flight   :  fuel_eff ~ 0.89
#   Poor flight (no crash) :  -0.5
#   Crash                  :  -1.0
# ==========================================
 
class LLSafetyFirst(UtilityFunction):
    """
    Three-level safety-first utility for Lunar Lander.
 
    u(r) = crash_penalty          if terminal == -100      (crashed)
           poor_flight_penalty    if shaping_norm < threshold  (flew poorly)
           fuel_eff               otherwise                (flew well)
 
    where shaping_norm = shaping / 200
          fuel_eff      = clip(-fuel / 35.8, 0, 1.5)
 
    Parameters
    ----------
    safety_threshold : float
        Minimum normalised shaping for an episode to be considered safe.
        Default 0.0 — any positive net shaping clears the bar.
        Episodes below this are penalised but not as harshly as crashes.
    poor_flight_penalty : float
        Penalty for episodes that didn't crash but flew poorly.
        Default -0.5, between crash_penalty and the fuel_eff range.
    crash_penalty : float
        Penalty for crash episodes. Default -1.0.
    """
 
    def __init__(
        self,
        safety_threshold: float  = 0.0,
        poor_flight_penalty: float = -0.5,
        crash_penalty: float     = -1.0,
    ):
        self.safety_threshold    = safety_threshold
        self.poor_flight_penalty = poor_flight_penalty
        self.crash_penalty       = crash_penalty
 
    def __call__(self, r):
        shaping = _shaping_norm(r)
        fuel    = _fuel_eff(r)
        crashed = _crashed(r)
        flew_poorly = shaping < self.safety_threshold
 
        if isinstance(r, np.ndarray):
            return np.select(
                [crashed, flew_poorly],
                [self.crash_penalty, self.poor_flight_penalty],
                default=fuel,
            )
        else:
            crash_val = torch.full_like(fuel, self.crash_penalty)
            poor_val  = torch.full_like(fuel, self.poor_flight_penalty)
            return torch.where(
                crashed,
                crash_val,
                torch.where(flew_poorly, poor_val, fuel),
            )
 
 
 
# ==========================================
# Hopper Calibration Utility
#
# Used ONLY for initial training to calibrate reward ranges.
# Run for 500k-1M steps, then use the trained policy to
# measure R[0], R[2] distributions for real utility design.
#
# Reward vector (cumulative episode return):
#   r[..., 0] = x_velocity      (forward progress, higher = faster)
#   r[..., 1] = jump_height     (z distance, mostly ignored)
#   r[..., 2] = energy_cost     (always <= 0, control cost)
# ==========================================
 
 
class HopperLinearCalibration(UtilityFunction):
    """
    Simple linear utility for calibration training run.
 
    u(r) = R[0] / 100.0
 
    Just forward velocity, normalised loosely so values are
    in a reasonable range for the critic to learn.
    Ignore R[1] and R[2] for now.
 
    After training converges, run calibrate_hopper.py to
    measure the true R[0] and R[2] ranges of a competent policy.
    """
 
    def __call__(self, r):
        alive = r[..., 2] > -400
        return torch.where(alive, r[..., 0] + r[..., 1] + 1e-3 * r[..., 2], r[..., 2])


 
class HopperEfficiency(UtilityFunction):
    """
    Simple linear utility for calibration training run.
 
    u(r) = R[0] / 100.0
 
    Just forward velocity, normalised loosely so values are
    in a reasonable range for the critic to learn.
    Ignore R[1] and R[2] for now.
 
    After training converges, run calibrate_hopper.py to
    measure the true R[0] and R[2] ranges of a competent policy.
    """
 
    def __call__(self, r):
        return (r[..., 0] + r[..., 1]) / (1e-6 + abs(r[..., 2]))


 
class HopperProduct(UtilityFunction):
    """
    Simple linear utility for calibration training run.
 
    u(r) = R[0] / 100.0
 
    Just forward velocity, normalised loosely so values are
    in a reasonable range for the critic to learn.
    Ignore R[1] and R[2] for now.
 
    After training converges, run calibrate_hopper.py to
    measure the true R[0] and R[2] ranges of a competent policy.
    """
 
    def __call__(self, r):
        alive = r[..., 2] > -400
        return torch.where(alive, r[..., 0] * r[..., 1] + 1e-3 * r[..., 2], r[..., 2])
 
 
# ==========================================
# Multi-Objective Ant Utilities
#
# Input reward vector (cumulative episode return):
#   r[..., 0] = x_velocity   (forward/right progress, positive = moving right)
#   r[..., 1] = y_velocity   (lateral progress, positive = moving up the plane)
#   r[..., 2] = control_cost (always <= 0, cumulative action effort)
#
# NOTE: Healthy reward has been removed. The agent receives no per-step
# survival bonus — it must discover locomotion to get any reward at all.
# This creates the same exploration challenge as Hopper without alive bonus.
#
# KEY DESIGN PRINCIPLES
# ---------------------
# 1. All utilities are non-linear to expose the ESR/SER gap.
# 2. Utilities with threshold structure (like DST/Hopper) create a hard
#    per-episode requirement that SER cannot satisfy in expectation.
# 3. Utilities with product structure expose the covariance gap directly:
#    ESR requires E[u(R)] != u(E[R]) by Jensen's inequality.
# 4. The directional utility exploits Ant's unique 2D velocity structure —
#    unavailable in Hopper/Cheetah — making the SER failure mode
#    geometrically interpretable: SER ant wanders, ESR ant navigates.
#
# CALIBRATION NOTES
# -----------------
# Run AntLinearCalibration for 500k steps first to establish typical
# per-episode return ranges. From a competent policy on standard Ant-v5:
#   R_x per episode: ~100-500 (good forward policy)
#   R_y per episode: ~-100 to +100 (lateral variation)
#   R_ctrl per episode: ~-50 to -200 (depends on gait)
# Without healthy reward these ranges will be lower — calibrate first.
# ==========================================
 
# ==========================================
# AntLinearCalibration
#
# Used ONLY for initial training to calibrate reward ranges.
# Run for 500k-1M steps with this utility, then measure
# R_x, R_y, R_ctrl distributions to set threshold constants
# for the real utility functions below.
# ==========================================
 
class AntLinearCalibration(UtilityFunction):
    """
    Simple linear utility for calibration run.
 
    u(r) = R_x / 100.0
 
    Just forward velocity loosely normalised. After training
    converges, measure the true R_x, R_y, R_ctrl distributions
    to calibrate the threshold and product utilities below.
    """
    def __call__(self, r):
        if isinstance(r, np.ndarray):
            return r[..., 0] / 100.0
        return r[..., 0] / 100.0
 
 
# ==========================================
# U1 — Directional Navigation with Threshold
#
# "Move right. Not just fast — specifically right."
#
# u = -|speed|             if NOT moving predominantly in X direction
#      R_x + R_y + w*R_ctrl  otherwise
#
# The directional condition: x_velocity > |y_velocity|
# This requires the ant to move more in X than Y within each episode.
#
# WHY ESR/SER GAP IS LARGE HERE
# ------------------------------
# SER optimises E[R_x] + E[R_y] independently. A SER policy can satisfy
# this by moving predominantly in Y in some episodes and X in others —
# both expectations are positive, but the directionality constraint is
# never met within any single trajectory.
#
# ESR explicitly optimises P(R_x > |R_y| AND speed is high) per episode.
# The SER failure mode is visually interpretable: the SER ant wanders
# in random directions, the ESR ant consistently moves right.
#
# This is structurally identical to DST and Hopper threshold utilities
# but exploits Ant's unique 2D velocity space — no other standard
# MuJoCo environment has this property.
#
# Expected ranges (after calibration):
#   Good directional episode  : ~4-8
#   Non-directional episode   : ~-2 to -5
# ==========================================
 
class AntDirectional(UtilityFunction):
    """
    Directional navigation utility for MO-Ant.
 
    u(r) = -|R_x + R_y|          if R_x <= |R_y|  (not going right enough)
            R_x + R_y + w*R_ctrl  otherwise
 
    Parameters
    ----------
    ctrl_weight : float
        Weight on control cost. Default 1e-3 keeps it small but present.
    """
    def __init__(self, ctrl_weight: float = 1e-4):
        self.w = ctrl_weight
 
    def __call__(self, r):
        x    = r[..., 0]
        y    = r[..., 1]
        ctrl = r[..., 2]
 
        speed = x + y
        is_directional = x > y.abs() if isinstance(r, torch.Tensor) else (x > np.abs(y))
 
        reward = speed + self.w * ctrl
 
        if isinstance(r, np.ndarray):
            penalty = -np.abs(speed)
            return np.where(is_directional, reward, penalty)
        else:
            penalty = -torch.abs(speed)
            return torch.where(is_directional, reward, penalty)
 
 
# ==========================================
# U2 — Speed-Efficiency Product
#
# "Move fast AND efficiently. Both must happen in the same episode."
#
# u = sqrt(R_x^2 + R_y^2) * clamp(1 + w * R_ctrl, 0, 2)
#
# The product of unsigned speed and a normalised efficiency term.
# R_ctrl is negative, so (1 + w * R_ctrl) is in (0, 1) for reasonable
# control costs — clamped to avoid negative efficiency from extreme actions.
#
# WHY ESR/SER GAP IS LARGE HERE
# ------------------------------
# The product utility directly exposes Cov(speed, efficiency):
#   ESR - SER = Cov(speed, efficiency)
# Speed and efficiency are negatively correlated within episodes
# (fast gaits require more torque = higher control cost).
# SER optimises E[speed] * E[efficiency] and ignores this covariance.
# ESR requires E[speed * efficiency] — the policy must find a gait
# that is simultaneously fast AND cheap, not just fast on average.
#
# Analogous to FTN product utility but in physical locomotion.
#
# Expected ranges (after calibration):
#   Fast efficient episode  : ~3-6
#   Fast wasteful episode   : ~1-2
#   Slow efficient episode  : ~0.5-1
# ==========================================
 
class AntSpeedEfficiency(UtilityFunction):
    """
    Speed-efficiency product utility for MO-Ant.
 
    u(r) = speed * efficiency
 
    where speed      = sqrt(R_x^2 + R_y^2).clamp(min=0)
          efficiency = clamp(1.0 + ctrl_weight * R_ctrl, 0, 2)
 
    Parameters
    ----------
    ctrl_weight : float
        Scales control cost into efficiency term. Default 1e-2.
        Set based on calibration so median episode efficiency ~ 0.7.
    """
    def __init__(self, ctrl_weight: float = 1e-2):
        self.w = ctrl_weight
 
    def __call__(self, r):
        x    = r[..., 0]
        y    = r[..., 1]
        ctrl = r[..., 2]
 
        if isinstance(r, np.ndarray):
            speed      = np.sqrt(x**2 + y**2).clip(min=0)
            efficiency = np.clip(1.0 + self.w * ctrl, 0.0, 2.0)
        else:
            speed      = torch.sqrt(x**2 + y**2).clamp(min=0)
            efficiency = torch.clamp(1.0 + self.w * ctrl, min=0.0, max=2.0)
 
        return speed * efficiency
 
 
# ==========================================
# U3 — Multi-Task Navigation Portfolio
#
# Three simultaneous utility functions sharing one pool of trajectories.
# Designed to demonstrate counterfactual reuse on a physically meaningful
# continuous control task — analogous to FTN Multi-Utility but for Ant.
#
# Task 0: AntNavigateRight   — maximise X velocity, ignore Y
# Task 1: AntNavigateDiagonal — maximise X + Y jointly (45-degree heading)
# Task 2: AntNavigateEfficient — maximise X + Y subject to control threshold
#
# WHY REUSE MATTERS HERE
# ----------------------
# All three tasks involve the same physical gait — the ant must move.
# An episode where the ant moves right fast (Task 0) is also informative
# for Task 1 (it has high X+Y) and Task 2 (if control cost is low).
# Without reuse, each task trains independently on 1/3 of the budget.
# With IS-weighted reuse, each task learns from all trajectories.
#
# This is the argument from the FTN no-reuse ablation (−24.247 vs −4.106
# on penalty utility) replicated in continuous control.
# ==========================================
 
class AntNavigateRight(UtilityFunction):
    """
    Task 0: Navigate right — maximise X velocity.
 
    u(r) = R_x + w * R_ctrl
 
    Simple directional objective. High ESR/SER gap only if combined
    with threshold or product structure; here it serves as the
    near-linear anchor task in the multi-task portfolio.
    """
    def __init__(self, ctrl_weight: float = 1e-3):
        self.w = ctrl_weight
 
    def __call__(self, r):
        return r[..., 0] + self.w * r[..., 2]
 
 
class AntNavigateDiagonal(UtilityFunction):
    """
    Task 1: Navigate diagonally — maximise X + Y jointly.
 
    u(r) = R_x + R_y + w * R_ctrl  if (R_x + R_y) > threshold
           R_ctrl                   otherwise (went backwards overall)
 
    Threshold ensures the agent must make net positive progress —
    episodes where the ant moves backward in both axes get no credit.
    The per-episode threshold creates an ESR/SER gap: SER can average
    forward and backward episodes; ESR penalises backward episodes directly.
 
    Parameters
    ----------
    progress_threshold : float
        Minimum (R_x + R_y) per episode to be considered progress.
        Set based on calibration — default -20.0 is a loose bar.
    ctrl_weight : float
        Weight on control cost. Default 1e-3.
    """
    def __init__(self, progress_threshold: float = -20.0, ctrl_weight: float = 1e-3):
        self.threshold = progress_threshold
        self.w = ctrl_weight
 
    def __call__(self, r):
        x    = r[..., 0]
        y    = r[..., 1]
        ctrl = r[..., 2]* 1e-3
 
        speed    = x + y
        is_progress = speed > self.threshold
        reward   = speed + self.w * ctrl
 
        if isinstance(r, np.ndarray):
            return np.where(is_progress, reward, ctrl)
        else:
            return torch.where(is_progress, reward, ctrl)
 
 
class AntNavigateEfficient(UtilityFunction):
    """
    Task 2: Navigate efficiently — speed subject to control cost threshold.
 
    u(r) = R_x + R_y + w * R_ctrl  if R_ctrl > ctrl_threshold
           R_ctrl                   otherwise (too expensive)
 
    The control threshold creates a hard per-episode efficiency requirement.
    SER can average expensive and cheap episodes; ESR penalises every
    expensive episode, forcing the policy to find efficient gaits consistently.
 
    Parameters
    ----------
    ctrl_threshold : float
        Maximum (most negative) allowable control cost per episode.
        Default -100.0 — episodes burning more than this get penalised.
        Set based on calibration of typical R_ctrl range.
    ctrl_weight : float
        Weight on control cost in the reward case. Default 1e-3.
    """
    def __init__(self, ctrl_threshold: float = -100.0, ctrl_weight: float = 1e-3):
        self.threshold = ctrl_threshold
        self.w = ctrl_weight
 
    def __call__(self, r):
        x    = r[..., 0]
        y    = r[..., 1]
        ctrl = r[..., 2]* 1e-3
 
        is_efficient = ctrl > self.threshold
        reward = x + y + self.w * ctrl
 
        if isinstance(r, np.ndarray):
            return np.where(is_efficient, reward, ctrl)
        else:
            return torch.where(is_efficient, reward, ctrl)
 
 
# ==========================================
# Convenience: collect the multi-task portfolio
# ==========================================
 
def get_ant_multitask_utilities(
    progress_threshold: float = -20.0,
    ctrl_threshold: float = -12.0,
    ctrl_weight: float = 1e-4,
):
    """
    Returns the three-task portfolio for multi-utility ESR-PPO training.
 
    Usage:
        utility_fns = get_ant_multitask_utilities()
        ppo = PPO(agent, optimizer, envs, utility_functions=utility_fns, ...)
 
    Parameters
    ----------
    progress_threshold : float
        Passed to AntNavigateDiagonal. Set after calibration.
    ctrl_threshold : float
        Passed to AntNavigateEfficient. Set after calibration.
    ctrl_weight : float
        Control cost weight shared across all three tasks.
    """
    return [
        AntNavigateRight(ctrl_weight=ctrl_weight),
        AntNavigateDiagonal(progress_threshold=progress_threshold,
                            ctrl_weight=ctrl_weight),
        AntNavigateEfficient(ctrl_threshold=ctrl_threshold,
                             ctrl_weight=ctrl_weight),
    ]
 