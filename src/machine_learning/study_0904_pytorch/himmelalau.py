import numpy as np
import matplotlib.pyplot as plt
import torch


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


# x = np.arange(-6,6,0.1)
# y = np.arange(-6,6,0.1)
#
# X,Y = np.meshgrid(x,y)
#
# z = himmelblau([X,Y])
# fig = plt.figure('himmelblau')
# ax = fig.gca(projection= '3d')
# ax.plot_surface(X,Y,z)
# ax.view_init(-60,30)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
#
# plt.show()
if __name__ == '__main__':
    x = torch.tensor([-4., 0.], requires_grad=True)
    # x' = x- lr * (df/dx) y' = y - lr * (df/dy)
    optimizer = torch.optim.Adam([x], lr=1e-3)

    for step in range(20000):
        pred = himmelblau(x)
        # 梯度清零
        optimizer.zero_grad()
        # 计算梯度
        pred.backward()
        # 更新 x,y
        optimizer.step()
        if step % 2000 == 0:
            print('step {} : x = {},f(x) = {}'.format(step, x.tolist(), pred.item()))
