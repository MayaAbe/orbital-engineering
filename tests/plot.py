import matplotlib.pyplot as plt


def plot2d(sol, color=None, title=None, xlabel=None, ylabel=None, aspectequal=True):
    if color is None:
        plt.plot(sol[:, 0], sol[:, 1])
    else:
        plt.plot(sol[:, 0], sol[:, 1], color)
    plt.grid()
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if aspectequal is True:
        plt.gca().set_aspect('equal')
    plt.show()


def plot3d(sol, color, title=None, xlabel=None, ylabel=None, zlabel=None, aspectequal=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if color is None:
        ax.plot(sol[:, 0], sol[:, 1], sol[:, 2])
    else:
        ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], color)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if zlabel is not None:
        ax.set_zlabel(zlabel)
    if aspectequal is True:
        ax.set_aspect('equal')
    plt.grid()
    plt.show()