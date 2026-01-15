import cmath
import warnings
from typing import Tuple, Dict, Union
from fractions import Fraction  # for denominator check

__all__ = ['aqueeq_binomial']

def aqueeq_binomial(
    x: complex,
    y: complex,
    alpha: complex,
    tol: float = 1e-12,
    max_terms: int = 100,
    boundary_threshold: float = 0.90,
    return_info: bool = False
) -> Union[complex, Tuple[complex, Dict]]:
    """
    Compute (x + y)^α using Aqueeq's Preconditioned Binomial Series Algorithm.
    Handles real negative bases with real fractional exponents preferring real odd roots.
    """
    # Trivial zero-input cases
    if x == 0 and y != 0:
        result = y ** alpha
        if return_info:
            return result, {"terms_used": 0, "used_fallback": False, "z": 0j}
        return result

    if y == 0 and x != 0:
        # Special: real negative base + real fractional exponent → prefer real root
        if x.imag == 0 and alpha.imag == 0 and x.real < 0:
            try:
                frac = Fraction(alpha.real).limit_denominator(1000)
                if frac.denominator % 2 == 1:  # odd root
                    abs_root = abs(x.real) ** alpha.real
                    result = complex(-abs_root)
                    if return_info:
                        return result, {"terms_used": 0, "used_fallback": False, "z": 0j, "note": "real odd root"}
                    return result
            except (ValueError, ZeroDivisionError):
                pass  # fall through to normal complex power

        result = x ** alpha
        if return_info:
            return result, {"terms_used": 0, "used_fallback": False, "z": 0j}
        return result

    # x = -y and integer alpha
    if x == -y and alpha.imag == 0 and alpha.real.is_integer():
        result = 0.0 + 0j
        if return_info:
            return result, {"terms_used": 0, "used_fallback": False, "z": None}
        return result

    # Zero sum
    if x + y == 0:
        if alpha.real > 0 and alpha.imag == 0:
            result = 0.0 + 0j
        else:
            raise ValueError("Cannot reliably compute 0 raised to non-positive or complex power")
        if return_info:
            return result, {"terms_used": 0, "used_fallback": True, "z": None}
        return result

    # Preconditioning
    ax, ay = abs(x), abs(y)
    if ax < ay:
        A = y
        z = x / y
    elif ax > ay:
        A = x
        z = y / x
    else:
        warnings.warn(f"|x| == |y| → log-exp fallback", RuntimeWarning)
        result = cmath.exp(alpha * cmath.log(x + y))
        if return_info:
            return result, {"terms_used": 0, "used_fallback": True, "z": None}
        return result

    # Boundary fallback
    used_fallback = False
    if abs(z) > boundary_threshold:
        warnings.warn(f"|z| = {abs(z):.4f} > {boundary_threshold} → log-exp", RuntimeWarning)
        result = cmath.exp(alpha * cmath.log(x + y))
        used_fallback = True
        if return_info:
            return result, {"terms_used": 0, "used_fallback": True, "z": z}
        return result

    # Series summation
    S = 1.0 + 0j
    term = 1.0 + 0j
    k = 1

    effective_max = max_terms
    if 0.75 < abs(z) <= boundary_threshold:
        effective_max = max(max_terms, int(max_terms * 3))

    while k <= effective_max:
        multiplier = (alpha - k + 1) * z / k
        term *= multiplier

        if abs(term) < 1e-16 * abs(S) or abs(term) < 1e-14:
            break

        S += term

        if abs(term) < tol * abs(S):
            break

        k += 1

    if k >= effective_max and abs(term) >= tol * abs(S):
        warnings.warn(
            f"No convergence after {effective_max} terms (|z|={abs(z):.4f}, |term|={abs(term):.2e})",
            RuntimeWarning
        )

    A_power = cmath.exp(alpha * cmath.log(A))
    result = A_power * S

    if return_info:
        info = {
            "terms_used": k,
            "last_term_abs": abs(term),
            "converged": abs(term) < tol * abs(S),
            "used_fallback": used_fallback,
            "z": z,
            "base": A
        }
        return result, info

    return result


# Tests (run this block)
if __name__ == "__main__":
    import numpy as np

    print("=== Aqueeq's Preconditioned Algorithm Examples ===\n")

    print(f"(1 + 8)^{{1/3}}     = {aqueeq_binomial(1, 8, 1/3).real:.12f}")
    print(f"(10000 + 62500)^{{0.5}} = {aqueeq_binomial(100**2, 250**2, 0.5):.10f}")
    print(f"10^{{0.75}}          = {aqueeq_binomial(10, 0, 0.75).real:.10f}")
    print(f"(1+2j + 3-1j)^{{0.5+0.5j}} = {aqueeq_binomial(1+2j, 3-1j, 0.5+0.5j)}")

    res, info = aqueeq_binomial(1, 1.1, 1/3, return_info=True)
    print(f"\nNear-boundary: {res.real:.10f}   Info: {info}")

    print(f"(2+3j -2-3j)^{{0.5}} = {aqueeq_binomial(2+3j, -2-3j, 0.5)}")

    # The fixed test
    res = aqueeq_binomial(-8, 0, 1/3)
    print(f"(-8)^{{1/3}} = {res.real:.10f}  (expected -2.0000000000)")
