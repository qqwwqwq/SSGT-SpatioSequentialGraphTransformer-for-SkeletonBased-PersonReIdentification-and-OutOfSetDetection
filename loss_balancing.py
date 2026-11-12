"""Dynamic loss balancing for Algorithm 2 of the SSGT paper.

The published rule uses a signed slope ratio, fixed increments, and a doubling
update period. The paper does not specify a finite-difference discretization;
this implementation retains the research code's central normalized difference.
Clamping the weight to [0, 1] keeps the two losses a convex combination.
"""

import math


PAPER_PROTOCOL = {
    'algorithm': 'SSGT paper Algorithm 2',
    'version': 'paper_algorithm_2_v1',
    'initial_lambda': 0.5,
    'initial_update_period': 10,
    'iota': 10.0,
    'tau': 0.05,
    'ratio': 'signed v_GPC / v_STPR',
    'update_schedule': 'absolute completed-epoch counts 10, 20, 40, 80, ...',
    'normalized_slope': '(L[i-1] - L[i+1]) / (L[i-1] + L[i+1])',
    'slope_discretization': 'implementation convention; not specified by the paper',
    'lambda_bounds': [0.0, 1.0],
    'boundary_handling': 'clip to [0, 1] to retain a convex loss combination',
    'zero_handling': 'no change for two zero slopes; signed epsilon for zero STPR slope',
    'convergence_lr_adjustments': False,
}


def mean_normalized_slope(losses, epsilon=1e-12):
    """Estimate the signed relative decline of a nonnegative loss history.

    Args:
        losses (sequence[float]): Chronological, finite, nonnegative losses.
        epsilon (float): Positive denominator floor for zero-valued losses.

    Returns:
        float: Mean central normalized difference. With two observations an
            adjacent difference is used; fewer observations give zero.
    """
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    values = [float(value) for value in losses]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('Loss history must contain finite, nonnegative values')
    if len(values) < 2:
        return 0.0
    pairs = [(values[0], values[1])] if len(values) == 2 else zip(values[:-2], values[2:])
    differences = [(before - after) / max(before + after, epsilon)
                   for before, after in pairs]
    return sum(differences) / len(differences)


def update_lambda(value, gpc_slope, stpr_slope, iota=10.0, tau=0.05, epsilon=1e-12):
    """Apply the signed-ratio, fixed-step update in Algorithm 2.

    Returns:
        tuple[float, float]: Clamped loss weight and signed slope ratio.
            Both zero slopes yield a neutral ratio of one.
    """
    values = (value, gpc_slope, stpr_slope, iota, tau, epsilon)
    if not all(math.isfinite(float(item)) for item in values):
        raise ValueError('Loss-balancing parameters must be finite')
    if not 0 <= value <= 1 or iota <= 1 or tau <= 0 or epsilon <= 0:
        raise ValueError('Require lambda in [0, 1], iota > 1, tau > 0, and epsilon > 0')
    if abs(gpc_slope) <= epsilon and abs(stpr_slope) <= epsilon:
        return float(value), 1.0
    denominator = (stpr_slope if abs(stpr_slope) > epsilon
                   else math.copysign(epsilon, stpr_slope))
    ratio = gpc_slope / denominator
    if ratio < 1.0 / iota:
        value += tau
    elif ratio > iota:
        value -= tau
    return min(1.0, max(0.0, float(value))), float(ratio)


class DynamicLossBalancer:
    """Track loss histories and update after 10, 20, 40, ... completed epochs.

    The interval value is an absolute update threshold, as in the paper's
    comparison of n against Up. Doubling therefore gives 10/20/40/80 rather
    than adding intervals to produce 10/30/70.
    """

    def __init__(self, initial_lambda=0.5, initial_period=10, iota=10.0, tau=0.05):
        if int(initial_period) != initial_period or initial_period < 2:
            raise ValueError('initial_period must be an integer of at least two')
        update_lambda(initial_lambda, 0.0, 0.0, iota, tau)
        self.initial_lambda = float(initial_lambda)
        self.initial_period = int(initial_period)
        self.value = float(initial_lambda)
        self.period = int(initial_period)
        self.iota = float(iota)
        self.tau = float(tau)
        self.gpc_losses = []
        self.stpr_losses = []

    def observe(self, gpc_loss, stpr_loss):
        """Record one epoch and return update details when its threshold is met."""
        # Validate before changing the history so failures leave state intact.
        mean_normalized_slope([gpc_loss, stpr_loss])
        self.gpc_losses.append(float(gpc_loss))
        self.stpr_losses.append(float(stpr_loss))
        epoch = len(self.gpc_losses)
        if epoch < self.period:
            return None
        gpc_slope = mean_normalized_slope(self.gpc_losses[-self.period:])
        stpr_slope = mean_normalized_slope(self.stpr_losses[-self.period:])
        previous = self.value
        self.value, ratio = update_lambda(self.value, gpc_slope, stpr_slope, self.iota, self.tau)
        event = {
            'completed_epochs': epoch,
            'period': self.period,
            'gpc_slope': gpc_slope,
            'stpr_slope': stpr_slope,
            'ratio': ratio,
            'lambda_before': previous,
            'lambda_after': self.value,
        }
        self.period *= 2
        event['next_update_epoch'] = self.period
        return event

    def metadata(self):
        """Describe the actual schedule without claiming unverified results."""
        description = dict(PAPER_PROTOCOL)
        description.update({
            'initial_lambda': self.initial_lambda,
            'initial_update_period': self.initial_period,
            'iota': self.iota,
            'tau': self.tau,
        })
        return description
