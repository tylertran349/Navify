import matplotlib.pyplot as plt

# Define the matrix
matrix = [
    [0.1, 0.2, 0.4, 0.3],
    [0.3, 0.1, 0.5, 0.1],
    [0.1, 0.2, 0.1, 0.6],
    [0.3, 0.3, 0.3, 0.1]
]

# Convert each row of the matrix into a string
# We'll use fixed-width spacing to keep columns aligned.
row_strings = []
for row in matrix:
    # Format each value with fixed width for alignment (e.g. width 4)
    formatted_values = " ".join(f"{val:0.1f}" for val in row)
    row_strings.append(formatted_values)

# Add the brackets around the entire matrix
# A neat way is to align them so it looks like a matrix:
# [ 0.1 0.3 0.3 0.3
#   0.3 0.1 0.3 0.3
#   0.3 0.3 0.1 0.3
#   0.3 0.3 0.3 0.1 ]
#
max_length = max(len(r) for r in row_strings)
matrix_str = "[ " + row_strings[0] + "\n"
for r in row_strings[1:-1]:
    matrix_str += "  " + r + "\n"
matrix_str += "  " + row_strings[-1] + " ]"

# Plot this text using matplotlib
fig, ax = plt.subplots(figsize=(3, 3))
ax.axis('off')

# Using a monospace font helps alignment look better
# Try 'Courier New' or another common monospace font
fontdict = {'family': 'monospace', 'size': 14}

# Place the text in the center
ax.text(0.5, 0.5, matrix_str, ha='center', va='center', fontdict=fontdict)

# Save as PNG with transparent background
plt.savefig("matrix.png", dpi=300, transparent=True)
plt.close()
