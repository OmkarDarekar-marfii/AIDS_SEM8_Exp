#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# Function to perform Union operation on fuzzy sets
def fuzzy_union(A, B):
    return np.maximum(A, B)

# Function to perform Intersection operation on fuzzy sets
def fuzzy_intersection(A, B):
    return np.minimum(A, B)

# Function to perform Complement operation on a fuzzy set
def fuzzy_complement(A):
    return 1 - A

# Function to perform Difference operation on fuzzy sets
def fuzzy_difference(A, B):
    return np.maximum(A, 1 - B)

# Function to create fuzzy relation by Cartesian product of two fuzzy sets
def cartesian_product(A, B):
    return np.outer(A, B)

# Function to perform Max-Min composition on two fuzzy relations (matrices)
def max_min_composition(R, S):
    R = np.atleast_2d(R)
    S = np.atleast_2d(S)
    result = np.zeros((R.shape[0], S.shape[1]))
    for i in range(R.shape[0]):
        for j in range(S.shape[1]):
            result[i, j] = np.max(np.minimum(R[i, :], S[:, j]))
    return result

# Example usage
A = np.array([0.2, 0.4, 0.6, 0.8])  # Fuzzy set A
B = np.array([0.3, 0.5, 0.7, 0.9])  # Fuzzy set B

# Operations on fuzzy sets
union_result = fuzzy_union(A, B)
intersection_result = fuzzy_intersection(A, B)
complement_A = fuzzy_complement(A)
difference_result = fuzzy_difference(A, B)

print("Union:", union_result)
print("Intersection:", intersection_result)
print("Complement of A:", complement_A)
print("Difference:", difference_result)

# Fuzzy relations
R = np.array([0.2, 0.5, 0.4])
S = np.array([0.6, 0.3, 0.7])

# Cartesian product of fuzzy relations
cartesian_result = cartesian_product(R, S)
print("Cartesian product of R and S:")
print(cartesian_result)

# To do max-min composition, we need 2 matrices
R_matrix = np.array([[0.2, 0.5, 0.4]])
S_matrix = np.array([[0.6, 0.3, 0.7]]).T

composition_result = max_min_composition(R_matrix, S_matrix)
print("Max-Min composition of R and S:")
print(composition_result)


# In[ ]:




