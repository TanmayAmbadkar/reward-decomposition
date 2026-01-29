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


        