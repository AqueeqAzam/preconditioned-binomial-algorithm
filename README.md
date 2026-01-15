# Aqueeq's Preconditioned Algorithm

Robust Python implementation for evaluating (x + y)^α using Newton's generalized binomial series with automatic preconditioning, log-exp fallback, and correct real negative roots.

## Installation

```bash
git clone https://github.com/GeniusAqueeq/aqueeq-binomial-algorithm.git
cd aqueeq-binomial-algorithm
pip install -r requirements.txt  # numpy if needed


Quick Example

from aqueeq_binomial import aqueeq_binomial

print(aqueeq_binomial(1, 8, 1/3))          # ≈ 2.080083823051904
print(aqueeq_binomial(-8, 0, 1/3))         # -2.0 (real cube root)
