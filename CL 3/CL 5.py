#!/usr/bin/env python
# coding: utf-8

# In[4]:


import random
import numpy as np

# Problem: maximize f(x) = 1 / (1 + x^2)  → highest at x = 0

POP_SIZE = 10
CLONE_FACTOR = 3
MUTATION_RATE = 0.2
GENERATIONS = 10

# Generate random antibody
def random_antibody():
    return random.uniform(-10, 10)

# Affinity function (higher is better)
def affinity(x):
    return 1 / (1 + x ** 2)

# Mutate with small random noise
def mutate(x):
    if random.random() < MUTATION_RATE:
        x += random.uniform(-1, 1)
    return x

def clonal_selection():
    population = [random_antibody() for _ in range(POP_SIZE)]
    best = None
    best_aff = 0

    for gen in range(GENERATIONS):
        affinities = [affinity(x) for x in population]
        sorted_pop = [x for _, x in sorted(zip(affinities, population), reverse=True)]

        print(f"Gen {gen+1}: Best Antibody = {round(sorted_pop[0], 4)} | Affinity = {round(affinity(sorted_pop[0]), 4)}")

        if affinity(sorted_pop[0]) > best_aff:
            best_aff = affinity(sorted_pop[0])
            best = sorted_pop[0]

        # Select top 3 for cloning
        selected = sorted_pop[:3]
        clones = []
        for antibody in selected:
            for _ in range(CLONE_FACTOR):
                clone = mutate(antibody)
                clones.append(clone)

        # Evaluate new population
        combined = selected + clones
        combined.sort(key=affinity, reverse=True)
        population = combined[:POP_SIZE]  # keep top individuals

    print(f"\n Final Best Antibody: {round(best, 4)} | Affinity: {round(best_aff, 4)}")

clonal_selection()


# In[ ]:





# In[ ]:





# In[ ]:




