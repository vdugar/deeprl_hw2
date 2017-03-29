import csv
import numpy as np
import matplotlib.pyplot as plt

# perf plot
dir_name = '/home/vdugar/courses/deep_rl/data/'
net_names = ['dqn', 'ddqn', 'duel', 'linear_naive', 'linear_soph', 'linear_double']
legend_key = ['DQN', 'Double DQN', 'Dueling DQN', 'Naive Linear', 'Linear', 'Linear Double']


fig1 = plt.figure()
for name in net_names:
    file = open(dir_name + name + '/results_' + name + '.txt', 'r')
    reader = csv.reader(file)
    yvals = []
    xvals = []
    for row in reader:
        xvals.append(row[0])
        yvals.append(row[2])

    file.close()

    plt.plot(xvals, yvals, linewidth=4)

plt.xlabel('Steps')
plt.ylabel('Avg. score over 20 episodes')
plt.legend(legend_key, loc=2)

plt.show()

# STD and mean for all final runs
for name in net_names:
    file = open(dir_name + name + '/final_' + name + '.txt', 'r')
    reader = csv.reader(file)
    yvals = []
    xvals = []
    for row in reader:
        xvals.append(row[0])
        yvals.append(row[1])
    file.close()

    a = np.float32(yvals)
    print("Mean: %f, STD: %f" % (np.mean(a), np.std(a)))