import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# Set up a grid for evaluation
x = np.linspace(-3, 6, 100)
y = np.linspace(-3, 6, 100)
X, Y = np.meshgrid(x, y)

# Define two bivariate normal distributions
mean1 = [0, 0]
cov1 = [[1, 0.5],
        [0.5, 1]]
rv1 = multivariate_normal(mean=mean1, cov=cov1)
Z1 = rv1.pdf(np.dstack((X, Y)))

mean2 = [2, 1]
cov2 = [[1, -0.3],
        [-0.3, 1]]
rv2 = multivariate_normal(mean=mean2, cov=cov2)
Z2 = rv2.pdf(np.dstack((X, Y)))

# Sum of the two distributions
Z_sum = Z1 + Z2

# Common colormap
cmap = 'viridis'

def style_3d_axis(ax):
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Remove axis panes and lines
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    ax.xaxis.line.set_color((0.0,0.0,0.0,0.0))
    ax.yaxis.line.set_color((0.0,0.0,0.0,0.0))
    ax.zaxis.line.set_color((0.0,0.0,0.0,0.0))
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    # Remove titles if any
    ax.set_title('')

# Plot and save Distribution 1
fig1 = plt.figure(figsize=(6,6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, Z1, cmap=cmap, edgecolor='none')
style_3d_axis(ax1)
plt.savefig('distribution1.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close(fig1)

# Plot and save Distribution 2
fig2 = plt.figure(figsize=(6,6))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X, Y, Z2, cmap=cmap, edgecolor='none')
style_3d_axis(ax2)
plt.savefig('distribution2.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close(fig2)

# Plot and save Sum of the Distributions
fig3 = plt.figure(figsize=(6,6))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(X, Y, Z_sum, cmap=cmap, edgecolor='none')
style_3d_axis(ax3)
plt.savefig('distribution_sum.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close(fig3)
