import numpy as np


successes = 0
trials = 1000000

for i in range(trials):
    if np.random.binomial(297, 0.01) >= 1:
        successes += 1

print(successes / trials)
