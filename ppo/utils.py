import numpy as np

class RunningMeanStd:
    """
    Calculates a running mean and standard deviation for a data stream.
    This is a numerically stable implementation using Welford's online algorithm.
    """
    def __init__(self, reward_size = 1):
        self.mean = np.zeros(reward_size)
        self.var = np.ones(reward_size)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        # Clip the standard deviation to prevent division by zero
        std = np.sqrt(self.var).clip(min=1e-8)
        return (x - self.mean) / std