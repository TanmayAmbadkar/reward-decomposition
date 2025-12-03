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
        if isinstance(r, np.ndarray):
            return np.prod(r, axis=-1)
        else:
            return torch.prod(r, dim=-1)

class FTNBenchmarkUtility(UtilityFunction):
    """
    The 5 utility functions used for Multi-Policy evaluation (MODeM 5.3.3).
    These functions are defined on 2 objectives (r0, r1).
    
    Modes:
    1: 'max' -> max(r0, r1)
    2: 'min' -> min(r0, r1)
    3: 'product' -> r0 * r1
    4: 'mixed' -> w0 * r0^2 + w1 * r1
    5: 'distance' -> - (w0(i0 - r0) + w1(i1 - r1))^2
    """
    def __init__(self, mode='max', w=(0.5, 0.5), ideal=(20.0, 20.0)):
        self.mode = mode
        self.w = w
        self.i = ideal

    def __call__(self, r):
        # We assume input is at least size 2. We use first two dims as per paper.
        r0 = r[..., 0]
        r1 = r[..., 1]

        if self.mode == 'max': # u1
            if isinstance(r, np.ndarray):
                return np.maximum(r0, r1)
            else:
                return torch.max(r0, r1)

        elif self.mode == 'min': # u2
            if isinstance(r, np.ndarray):
                return np.minimum(r0, r1)
            else:
                return torch.min(r0, r1)

        elif self.mode == 'product': # u3
            return r0 * r1

        elif self.mode == 'mixed': # u4: w0*r0^2 + w1*r1
            return self.w[0] * (r0 ** 2) + self.w[1] * r1

        elif self.mode == 'distance': # u5: Weighted distance to ideal point
            term0 = self.w[0] * (self.i[0] - r0)
            term1 = self.w[1] * (self.i[1] - r1)
            return -((term0 + term1) ** 2)

        else:
            raise ValueError(f"Unknown mode {self.mode}")

environment_map = {
    "deep-sea-treasure-1": [DSTDebtUtility(),],
    "deep-sea-treasure-2": [DSTGeneralUtility(mode='linear'), DSTGeneralUtility(mode='threshold'), DSTGeneralUtility(mode='ratio')],
    "fruit-tree-1": [FTNProductUtility()],
    "fruit-tree-5": [FTNBenchmarkUtility(mode='max'), FTNBenchmarkUtility(mode='min'), FTNBenchmarkUtility(mode='product'), FTNBenchmarkUtility(mode='mixed'), FTNBenchmarkUtility(mode='distance')],
}