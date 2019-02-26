import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.colors import BoundaryNorm
# from matplotlib.ticker import MaxNLocator
import numpy as np
# import math
import pandas as pd
# from pandas import ExcelWriter
# from pandas import ExcelFile


z = pd.read_excel('/home/williamwoo/Desktop/Associated_data4_A_block_one_curve_one_rectangle.xlsx', sheet_name='Sheet3')

x = z ['pos_x']
y = z ['pos_y']

u = z ['mag_x']
v = z ['mag_y']
w = z ['mag_z']


# X, Y =np.meshgrid(x, y)

plt.figure(figsize=(6, 9))
plt.gca().set_aspect('equal')

# m = plt.contourf(X, Y, Z, cmap=plt.cm.jet)
# m = plt.scatter(x, y, c = u/v/w, cmap='jet')                  #for x/y/z direction respetively
m = plt.scatter(x, y, c = np.sqrt(u**2+v**2+w**2), cmap='jet')

# cbar=plt.colorbar(m,orientation='vertical')                               
# cbar.set_label('Magnetic Field Strength ($\mu$T)', rotation=270)

n = plt.colorbar(m)                               
n.set_label('Magnetic Field Strength ($\mu$T)', rotation=270)

plt.title('Magnetic Field Map')
plt.xlabel('$Direction  X$ (m)')
plt.ylabel('$Direction  Y$ (m)')


plt.show()

