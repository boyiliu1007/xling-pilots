import math


def wilson(k, n, z=1.959963984540054):
    """95% Wilson score interval. None if n == 0."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    """'k/n = 0.xxx [lo, hi]' or 'k/0 = N/A'."""
    if n == 0:
        return f"{k}/{n} = N/A"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]"
