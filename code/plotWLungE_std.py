import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.special import sph_harm
from numpy import loadtxt
import sys
        
name_out = sys.argv[1]

phi = np.linspace(0, np.pi, 100)
theta = np.linspace(0, 2*np.pi, 100)
phi, theta = np.meshgrid(phi, theta)

n = 100-1 # 100 is number of quadorder in julia script
phi = [(k+0.5)*np.pi/n for k in range(2*(n+1))]
mu, w = np.polynomial.legendre.leggauss(n+1)
mu = mu.T
print(len(mu))
print(len(phi))
mu, phi = np.meshgrid(mu, phi)

# The Cartesian coordinates of the unit sphere
x = np.sqrt(1.0 - mu*mu) * np.cos(phi)
y = np.sqrt(1.0 - mu*mu) * np.sin(phi)
z = mu

m, l = 2, 3

# Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
fcolors = sph_harm(m, l, mu, phi).real
fmax, fmin = fcolors.max(), fcolors.min()
fcolors = (fcolors - fmin)/(fmax - fmin)

fcolors = loadtxt("output/W_plot_lung.txt", comments="#", dtype = float, unpack=False)
fcolors1 = fcolors[:,0].reshape((2*n+2,n+1))
fmax, fmin = fcolors1.max(), fcolors1.min()
fcolors1 = (fcolors1 - fmin)/(fmax - fmin)
fcolors2 = fcolors[:,1].reshape((2*n+2,n+1))
fmax, fmin = fcolors2.max(), fcolors2.min()
fcolors2 = (fcolors2 - fmin)/(fmax - fmin)
fcolors3 = fcolors[:,2].reshape((2*n+2,n+1))
fmax, fmin = fcolors3.max(), fcolors3.min()
fcolors3 = (fcolors3 - fmin)/(fmax - fmin)
fcolors4 = fcolors[:,3].reshape((2*n+2,n+1))
fmax, fmin = fcolors4.max(), fcolors4.min()
fcolors4 = (fcolors4 - fmin)/(fmax - fmin)

# Set the aspect ratio to 1 so our sphere looks spherical
fig = plt.figure(figsize=(15,13))
ax = fig.add_subplot(221, projection='3d')
ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.plasma(fcolors1), antialiased=True)
ax.view_init(-80, -90)
ax.set_title(r"E[$W_1$]", fontsize=20)
ax.set_xlabel("y", fontsize=15)
ax.set_ylabel("z", fontsize=15)
ax.set_zlabel("x", fontsize=15)
ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_zticks([-1.0, 0.0, 1.0])
ax.tick_params("both",labelsize=13) 
# Turn off the axis planes
#ax.set_axis_off()
ax = fig.add_subplot(222, projection='3d')
ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.plasma(fcolors2), antialiased=True)
ax.view_init(-80, -90)
ax.set_title(r"E[$W_2$]", fontsize=20)
ax.set_xlabel("y", fontsize=15)
ax.set_ylabel("z", fontsize=15)
ax.set_zlabel("x", fontsize=15)
ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_zticks([-1.0, 0.0, 1.0])
ax.tick_params("both",labelsize=13) 
ax = fig.add_subplot(223, projection='3d')
ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.plasma(fcolors3), antialiased=True)
ax.view_init(-80, -90)
ax.set_title(r"std($W_1$)", fontsize=20)
ax.set_xlabel("y", fontsize=15)
ax.set_ylabel("z", fontsize=15)
ax.set_zlabel("x", fontsize=15)
ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_zticks([-1.0, 0.0, 1.0])
ax.tick_params("both",labelsize=13) 
ax = fig.add_subplot(224, projection='3d')
ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.plasma(fcolors4), antialiased=True)
ax.view_init(80, 90)
ax.set_title(r"std($W_2$)", fontsize=20)
ax.set_xlabel("y", fontsize=15)
ax.set_ylabel("z", fontsize=15)
ax.set_zlabel("x", fontsize=15)
ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_zticks([-1.0, 0.0, 1.0])
ax.tick_params("both",labelsize=13) 
fig.tight_layout()
plt.savefig(name_out)
#plt.show()

 
