import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Random 8x8 attention scores (replace with your actual scores)
attention_scores = np.random.rand(8, 8)

# Create heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(attention_scores, annot=True, fmt=".2f", cmap='viridis', cbar=True)

plt.title("8x8 Attention Scores Heatmap")
plt.xlabel("Key positions")
plt.ylabel("Query positions")

plt.show()