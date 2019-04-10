import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from scipy import interpolate
from scipy.interpolate import griddata
import scipy.interpolate as interp



z = pd.read_excel('/home/williamwoo/Desktop/Mag_Map/B4-counterclock-interpolated-magplot.xlsx', sheet_name='Sheet1')

x = z ['pos_x']
y = z ['pos_y']

u = z ['mag_x']
v = z ['mag_y']
w = z ['mag_z']


# xx, yy = np.meshgrid(x, y)
# xx, yy = np.mgrid[0:57:1j,0:2:1j]
# xx, yy = np.meshgrid(np.arange(0, 57, 0.05), np.arange(0, 1.5, 0.05))
# points = (68166, 2)
# x, y = x.flatten(), y.flatten()
# d = np.sqrt(u**2+v**2+w**2)
# c = c.flatten()

plt.figure(figsize=(20, 10))
plt.gca().set_aspect('equal')

############################################ For interpolation:
# m = interpolate.interp2d(x, y, c, kind='cubic')


# m = griddata((x, y), d, (xx, yy), method='cubic')
# plt.imshow(m, extent=(0,58,-0.8,0.8), origin='lower', cmap='jet')


# m = plt.contourf(x, y, c, levels=100)
# m = interp.Rbf(x, y, d, function='cubic', smooth=0)
# plt.contourf(x, y, m, cmap='jet')

# triang = tri.Triangulation(x, y)
# interpolator = tri.LinearTriInterpolator(triang, d)
# dd = interpolator(xx, yy)
# plt.contour(xx, yy, dd, linewidths=0.5, colors='k')
# cntr1 = plt.contourf(xx, yy, dd, cmap="RdBu_r")
# plt.colorbar(cntr1)

# rbf = Rbf(x, y, d, function='cubic', smooth=0)
# zz = rbf(xx, yy)
# m = plt.pcolor(xx, yy, zz, cmap='jet')
# plt.colorbar(m)
###################################################################

m = plt.scatter(x, y, c = np.sqrt(u**2+v**2+w**2), cmap='jet')
# m = plt.scatter(x, y, c=u, cmap='jet')

################################################### below part for annotations:
# fig, ax = plt.subplots()
# ax.scatter(x, y)

# for i, txt1 in enumerate(u):
#    ax.annotate((txt1, txt2), (x[i], y[i]))

# for j, txt in enumerate(v):
#     ax.annotate(txt, (x[j], y[j]))

# label = (u, v)

# plt.annotate(label, # this is the text
#              (x,y), # this is the point to label
#             #  textcoords="offset points", # how to position the text
#              xytext=(0,10), # distance from text to points (x,y)
#              ha='center')
##############################################################################

# cbar=plt.colorbar(m,orientation='vertical')                               
# cbar.set_label('Magnetic Field Strength ($\mu$T)', rotation=270)

# plt.imshow(m(x, y, d), cmap='jet')
# plt.colorbar(shrink=0.6)

n = plt.colorbar(m, shrink=0.54)                               
n.set_label('Magnetic Field Strength ($\mu$T)', rotation=270, size=15, labelpad=30)


plt.title('Magnetic Field Map', size=20)
plt.xlabel('$Direction  X$ (m)', size=15)
plt.ylabel('$Direction  Y$ (m)', size=15)

plt.show()

