import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from scipy import interpolate
from scipy.interpolate import griddata
import scipy.interpolate as interp



z = pd.read_excel('/home/williamwoo/Desktop/Mag_Map/04-01-WKW-carpark-0.05-scaledown.xlsx', sheet_name='Sheet1')

x = z ['pos_x']
y = z ['pos_y']

u = z ['mag_x']
v = z ['mag_y']
w = z ['mag_z']

# c = z ['magnitude']

# u = z ['sqrt']
# v = z ['same_z']

U = u / np.sqrt(u**2 + v**2)
V = v / np.sqrt(u**2 + v**2)

# plt.figure()
plt.title('5% Dataset scaled-down, Directions of the X+Y Magnetic Field', size=20)
Q = plt.quiver(x, y, U, V, units='width', scale=40)
# Q = plt.quiver(x, y, U, V, units='width', scale=60)

# Q = plt.quiver(x, y, U, V, units='width')

# qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='figure')
plt.scatter(x, y, color='r', s=10)


# plt.figure(figsize=(10, 20))
# plt.figure(figsize=(20, 10))
plt.gca().set_aspect('equal')

# m = plt.scatter(x, y, c = np.sqrt(u**2+v**2+w**2), cmap='jet')
# m = plt.scatter(x, y, c, cmap='jet')


# n = plt.colorbar(m, shrink=0.54)                               
# n.set_label('Magnetic Field Strength ($\mu$T)', rotation=270, size=15, labelpad=30)


# plt.title('Magnetic Field Intensities', size=20)
# plt.xlabel('$Direction  X$ (m)', size=15)
# plt.ylabel('$Direction  Y$ (m)', size=15)

plt.show()