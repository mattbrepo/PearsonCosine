# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# Generate random values from 0 to 3
np.random.seed(42)
n = 100  # Size of the dataset
set1 = np.random.randint(0, 4, n)

# Define offset range
offset_range = np.arange(-50, 51, 1)
pearson_values = []
cosine_values = []

# Simulate correlation with different offsets
for offset in offset_range:
    set2 = set1 + offset
    
    # Compute Pearson correlation
    pearson_corr, _ = pearsonr(set1, set2)
    pearson_values.append(pearson_corr)
    
    # Compute Cosine similarity
    cosine_sim = cosine_similarity([set1], [set2])[0, 0]
    cosine_values.append(cosine_sim)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(offset_range, pearson_values, label='Pearson Correlation', linestyle='-', marker='o')
plt.plot(offset_range, cosine_values, label='Cosine Similarity', linestyle='-', marker='s')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Offset')
plt.ylabel('Correlation / Similarity')
plt.title('Pearson Correlation and Cosine Similarity vs Offset')
plt.legend()
plt.grid()
plt.show()
