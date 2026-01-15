# Aqueeq's Preconditioned Algorithm

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/GeniusAqueeq/aqueeq-binomial-algorithm?style=social)](https://github.com/GeniusAqueeq/aqueeq-binomial-algorithm)

**Robust, production-ready Python implementation** for evaluating \((x + y)^\alpha\) using Newton's generalized binomial series — with **automatic preconditioning**, **log-exp fallback**, and **correct real negative roots** for fractional powers.

### Why this library?

Most scientific computing libraries avoid the binomial series for non-integer exponents, preferring log-exp or built-in `pow` functions. However, the series form is valuable in cases such as:
- Automatic differentiation / AD tools
- Taylor series expansions
- Symbolic manipulation
- Resource-constrained environments (embedded systems, GPUs without fast pow)
- Educational / pedagogical purposes

Naive implementations often diverge or converge extremely slowly.  
This library solves that by **automatically preconditioning** the expansion and handling edge cases that almost everyone overlooks.

### Features

- **Guaranteed convergence** — never diverges like naive series
- **Automatic domain conditioning** — always chooses the larger-magnitude term as base → |z| < 1
- **Intelligent fallback** to `cmath.exp(alpha * cmath.log(x + y))` when |z| is too close to 1
- **Correct real roots for negative bases** — e.g. `(-8)^{1/3} = -2.0` (not Python's default +1.0 complex branch)
- **Full edge-case handling**:
  - x = 0 or y = 0
  - x = -y with integer α
  - x + y = 0
  - |x| ≈ |y| (slow convergence zone)
- **Diagnostics returned** (terms used, convergence status, |z|, fallback used, etc.)
- Clean, type-hinted, documented code with warnings and tests

### Installation

```bash
# Recommended: clone and use locally

from aqueeq_binomial import aqueeq_binomial

# Classic example: cube root of 9
print(aqueeq_binomial(1, 8, 1/3))           # ≈ 2.080083823051904 + 0j

# Real negative base → correct real root
print(aqueeq_binomial(-8, 0, 1/3))          # -2.0 + 0j (not 1.0!)

# Apparent power in power systems: √(P² + Q²)
print(aqueeq_binomial(100**2, 250**2, 0.5)) # ≈ 269.2582403567 + 0j

# Complex exponent
print(aqueeq_binomial(1+2j, 3-1j, 0.5+0.5j))
# → something like (1.211 + 1.327j)

# Get detailed info
res, info = aqueeq_binomial(1, 1.1, 1/3, return_info=True)
print(res, info)
# Shows terms_used=0, used_fallback=True, z=0.909..., converged=True, etc.
git clone https://github.com/GeniusAqueeq/aqueeq-binomial-algorithm.git
cd aqueeq-binomial-algorithm
pip install -r requirements.txt          # usually just numpy
