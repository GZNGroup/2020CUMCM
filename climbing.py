import numpy as  np
pointdata=np.random.randint(1,100,size=(1000,2))
print(pointdata)
score=np.random.randn(1000)
score=score.reshape(-1,1)
print(score)
finaldata=np.hstack((pointdata,score))
print(finaldata)

#实时输出坐标轴x,y
def output(x,y):
    return -x**2-y**2-x*y+4*x
'''
def output(x,y):
    score=finaldata(x,y)
    return score
'''

def test(x0,y0,fanx,fany):
    #定义初始值的大小
    point_x=x0
    point_y=y0

    step=0.001
    point_history=[]
    score_history=[]

    i=0
    while (point_x<fanx[1] and point_x>fanx[0]) and (point_y<fany[1] and point_y>fany[0]) and i<1e4:
        x0=point_x
        y0=point_y
        while point_x<fanx[1] and point_x>fanx[0]:

            last_point_x=point_x
            gradient = (output(point_x+step, point_y) - output(point_x, point_y)) / step
            point_x=point_x+step*np.sign(gradient)

            if abs(output(point_x,point_y)-output(last_point_x,point_y))<0.0001:
                print(point_x)
                point_history.append((point_x,point_y))
                score_history.append(output(point_x,point_y))
                break

        while point_y<fany[1] and point_y>fany[0]:

            last_point_y = point_y
            gradient = (output(point_x, point_y+step) - output(point_x, point_y)) / step
            point_y = point_y + step * np.sign(gradient)

            if abs(output(point_x,point_y)-output(point_x,last_point_y))<0.00001:
                point_history.append((point_x, point_y))
                score_history.append(output(point_x, point_y))
                print(point_y)
                break

        if abs(output(point_x, point_y) - output(x0, y0)) < 0.0000001 or abs(point_x-x0) < 0.00001:
            break
        i=i+1
    print(point_x, point_y,x0, y0)

test(2,8,[-10,10],[-10,10])

# 画出图像如下进行检查
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(10,6))
ax = Axes3D(fig)
x = np.arange(-10, 10, 0.5)
y = np.arange(-10, 10, 0.5)
X, Y = np.meshgrid(x, y)
Z =-X**2 - Y**2-X*Y+4*X
plt.xlabel('x')
plt.ylabel('y')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
print(Z)
print(np.max(Z))