import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


def plt_surface(fig, sheet_name, rows=1, cols=1, index=1):
    df = pd.read_excel(r'C:\Users\zhangxiaochuan-jk\Desktop\umtp test\params_robust..xlsx', sheet_name=sheet_name,
                       index_col=0).iloc[:39, :19]  # [40:78]  # [:39]  # [40:78]
    X = np.array([float(i.split(',')[0]) if isinstance(i, str) else i for i in df.columns.values], dtype=float)
    Y = np.array([float(i.split('_')[0]) if isinstance(i, str) else i for i in df.index.values], dtype=float)
    X, Y = np.meshgrid(X, Y)
    Z = df.values

    ax = fig.add_subplot(rows, cols, index, projection='3d')
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_title(sheet_name.split('_')[0])
    ax.set_xlabel(r'$\beta$', labelpad=-5)
    ax.set_ylabel(r'$\alpha$', labelpad=-5)
    ax.tick_params(axis="both", pad=-2)
    ax.view_init(azim=-15)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # fig.colorbar(surf, shrink=0.5, aspect=5)


# sheet_names = ['cora_web', 'citeseer_web', 'computers_web', 'photo_web', 'pubmed_web', 'cs_web', 'arxiv_web']
sheet_names = ['cora_web', 'computers_web', 'pubmed_web', 'cs_web']
rows, cols = 2, 2
fig = plt.figure(figsize=(12, 12), dpi=80)
# ax = fig.add_subplot(projection='3d')
for i in range(rows):
    for j in range(cols):
        if i*cols+j < len(sheet_names):
            plt_surface(fig, sheet_names[i*cols+j], rows, cols, i*cols+j+1)


plt.show()


if __name__ == '__main__':
    pass
