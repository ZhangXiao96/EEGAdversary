import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn
import os

seaborn.set_style('darkgrid')

np.random.seed(1000)

# ========================
subject = 'B'
model_name = 'xDAWN+Riemann+LR'
epsilon = 0.6
# =========================

model_path = os.path.join('..', 'runs', model_name, subject, 'attack_acc.npz')
target_char_list = list('abcdefghijklmnopqrstuvwxyz123456789_')

N = len(target_char_list)
acc_dict = np.load(model_path)['acc'].item()

radii = np.array([acc_dict[char] for char in target_char_list]) * 100

min_acc = np.min(radii)
if min_acc % 2 == 0:
    pad = 0
else:
    pad = 1
bottom = np.min(radii) - 4 + pad
y_lower_limit = bottom - 2 + pad
y_upper_limit = 100

x_space = 2 * np.pi / (N+1)
theta = np.linspace(0.0, 2 * np.pi, N + 1, endpoint=False)[1:] + x_space / 2.0
width = x_space * 0.5

plt.figure(figsize=(5, 5))
ax = plt.subplot(111, polar=True)
bars = ax.bar(theta, radii-bottom, width=width, bottom=bottom)

ax.xaxis.set_major_locator(ticker.MultipleLocator(x_space))
ax.xaxis.set_ticklabels([])

ax.set_ylim([y_lower_limit, y_upper_limit])
ax.set_yticks(np.arange(bottom+2, y_upper_limit+1, 2, ))
ax.set_rlabel_position(0)

# color = np.arange(0, 1., 1./N)
# np.random.shuffle(color)
colors = ['tomato', 'lightskyblue']
for i in range(N):
    bar = bars[i]
    r = radii[i]
    char = target_char_list[i]
    x = theta[i]
    # bar.set_facecolor(plt.cm.jet(color[i]))
    bar.set_facecolor(color=colors[i % 2])
    bar.set_alpha(0.8)
    ax.text(x, 101, char.upper(), verticalalignment='center', horizontalalignment='center', fontsize=13)

# plt.tight_layout(pad=1.02)
print()
print('Subject {} (epsilon={}):'.format(subject, epsilon))
print('Mean Successful Rate: {:.1f}%.'.format(np.mean(radii)))
plt.show()
