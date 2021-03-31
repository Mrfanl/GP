import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def gaussian_dist(X,mu=0.,sigma=1.):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5/(sigma**2)*(X-mu)**2)

def guassian_dist_2dim(X,Y,mu_x,sigma_x,mu_y,sigma_y,rho=0):
    return 1/(2*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2))*\
           np.exp(-1/(2-2*rho**2)*((X-mu_x)**2/sigma_x**2-2*rho*(X-mu_x)*(Y-mu_y)/(sigma_x*sigma_y)+\
               (Y-mu_y)**2/sigma_y**2))

# 一元高斯分布
# X = np.arange(-10,10,0.1)
# Y_0_1 = gaussian_dist(X)
# Y_0_2 = gaussian_dist(X,mu=0.,sigma=2)
# Y_3_2 = gaussian_dist(X,mu=3.,sigma=2)
# Y_0_10 = gaussian_dist(X,mu=0,sigma=10)

# plt.plot(X,Y_0_1,label='mu=0,sigma=1.')
# plt.plot(X,Y_0_2,label='mu=0,sigma=2.')
# plt.plot(X,Y_3_2,label='mu=3,sigma=2.')
# plt.plot(X,Y_0_10,label='mu=0,sigma=10.')

# plt.legend()
# plt.show()

# 二元高斯分布
X = np.arange(-10,10,0.1)
Y = np.arange(-10,10,0.1)
X,Y = np.meshgrid(X,Y)
Z1 = guassian_dist_2dim(X,Y,0,3,0,3,rho=0.)
fig = plt.figure()
ax = Axes3D(fig)

ax.plot_surface(X,Y,Z1,rstride=1, cstride=1, alpha=0.5,cmap=cm.coolwarm)
plt.legend()
plt.show()
