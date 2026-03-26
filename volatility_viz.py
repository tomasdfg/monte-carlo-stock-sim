import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde


plt.style.use("dark_background")

# simulation parameters
S0 = 100      # starting price (normalized)
mu = 0.12     # expected annual return
sigma = 0.25  # annual volatility
T = 2         # years to forecast
n = 100       # time steps
M = 1000      # number of simulations

dt = T/n

St = np.exp(
    (mu - sigma ** 2 / 2) * dt
    + sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T
)
St = np.vstack([np.ones(M), St])
St = S0 * St.cumprod(axis=0)

time = np.linspace(0,T,n+1)
time_points = np.linspace(0, n, 10, dtype=int)


price_range = np.linspace(St.min(), St.max(), 200)

densities = []
for t in time_points:
    kde = gaussian_kde(St[t])
    densities.append(kde(price_range))

densities = np.array(densities)
densities_normalized = densities / densities.max()

T_grid, P_grid = np.meshgrid(time_points, price_range)




fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')


percentiles = [5, 25, 50, 75, 95]
bands = np.percentile(St, percentiles, axis=1)

Y = np.array([0, 1, 2, 3, 4])
X, Y_grid = np.meshgrid(time, Y)


ax2.plot_surface(T_grid, P_grid, densities_normalized.T, cmap='RdYlGn', alpha=0.8)




ax.plot_surface(X, Y_grid, bands, alpha=0.7, cmap='RdYlGn')
ax.set_xlabel("Time (Years)")
ax.set_ylabel("percentile band")
ax.set_zlabel("price values")
ax.set_title("Monte Carlo Volatility Cone")

ax2.set_title("Price Probability Surface")
ax2.view_init(elev=20, azim=250)



plt.show()