import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)

x = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]
y = [2048, 2547.51, 3371.71, 2642.44, 3421.66, 3346.73,
     3396.68, 3596.49, 3796.29, 3796.29]
z1 = np.polyfit(x, y, 1)
p1 = np.poly1d(z1)
y_pre = p1(x)


plt.plot(x, y, label='实际值')
plt.plot(x, y_pre, label='拟合值')
plt.xlabel('松紧度(压力值)', fontproperties=font)
plt.ylabel('频率(HZ)', fontproperties=font)

# 在图中显示方程
equation = f'{p1}'
plt.text(0.6, 0.95, equation, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.show()
