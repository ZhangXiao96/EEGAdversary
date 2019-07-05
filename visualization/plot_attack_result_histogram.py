import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn
import os

# seaborn.set_style('darkgrid')

# ========================
subject = 'A'
model_name = 'xDAWN+Riemann+LR'
epsilon = 0.5
nb_rounds = 15
# =========================
if subject == 'A':
    down_limit = 0.85
    up_limit = 1.00
elif subject == 'B':
    down_limit = 0.8
    up_limit = 1.00
else:
    raise Exception('No subject named \'{}\''.format(subject))

bar_width = 0.6
bar_step = 0.2


model_path = os.path.join('..', 'runs', model_name, subject, 'attack_acc_{}.npz'.format(nb_rounds))
target_char_list = list('abcdefghijklmnopqrstuvwxyz123456789_')

N = len(target_char_list)
acc_dict = np.load(model_path)['acc'].item()

# plot
plt.figure(figsize=(8, 3))

score = np.array([acc_dict[char] for char in target_char_list])
x = range(len(score))

rects1 = plt.bar(x=x, height=score, width=bar_width, alpha=0.8, label="15 rounds")
plt.ylim(down_limit, up_limit)
plt.yticks(np.arange(down_limit, up_limit+0.01, 0.05))
plt.yticks(fontsize=12)
plt.ylabel("Attack Score", fontsize=14)

plt.xticks(x, list('abcdefghijklmnopqrstuvwxyz123456789_'.upper()), fontsize=11)
plt.xlabel("Target Character", fontsize=14)
plt.xlim([-0.5, len(target_char_list)-0.5])
# plt.legend(fontsize=12)
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + (rect.get_width()+bar_step) / 2, height+1, str(height), ha="center", va="bottom")

plt.tight_layout()
plt.show()

print()
print('Subject {} (epsilon={}):'.format(subject, epsilon))
print('Mean Successful Rate: {:.1f}%.'.format(np.mean(score*100)))
