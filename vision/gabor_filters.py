import numpy as np
from datavyz.main import graph_env


def gabor(x, y,
          sigma=1, theta=1, Lambda=1, psi=1, gamma=1):
    """
    Gabor feature extraction.
    """
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    # (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


ge = graph_env('visual_stim')

x, y = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))
z = gabor(x, y)

fig, _, _ = ge.twoD_plot(x.flatten(), y.flatten(), z.flatten(), colormap=ge.binary)

fig.savefig('fig.png')

# ge.show()
