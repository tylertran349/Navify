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
    
    ax.xaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    # Remove titles if any
    ax.set_title('')

# Function to create and save a plot
def create_and_save_plot(Z, filename):
    fig = plt.figure(figsize=(6, 6), facecolor='none')  # Set figure facecolor to 'none' for transparency
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none')
    style_3d_axis(ax)
    
    # Make the background of the axes transparent
    ax.patch.set_alpha(0.0)
    
    # Save the figure with transparent background
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
    plt.close(fig)

# Plot and save Distribution 1
create_and_save_plot(Z1, 'distribution1.png')

# Plot and save Distribution 2
create_and_save_plot(Z2, 'distribution2.png')

# Plot and save Sum of the Distributions
create_and_save_plot(Z_sum, 'distribution_sum.png')
