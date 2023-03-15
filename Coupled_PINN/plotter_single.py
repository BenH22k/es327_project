import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import csv
import numpy as np


if True:

    results = []

    with open('Coupled_PINN/results/RESULTS1.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            newrow = []
            for elem in row:
                newrow.append(np.float32(elem))

            results.append(newrow)

    t_test = results[0]
    x2_test = results[1]
    x2 = results[2]
    #x2_I = results[3]
        

if True:
    subtitle_fontsize = 18
    x_scaling = 1
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(14,7)
    fig.set_dpi(300)
    #fig.suptitle('Coupled Oscillator - tanh vs ReLU, $d$',fontsize=26)
    #for i in range(2):
    for j in range(2):
        axs[j].grid(True)
        axs[j].set_xlabel("t",fontsize=16)
        axs[j].set_ylabel("$x_2$",fontsize=16)
        axs[j].yaxis.set_major_locator(MaxNLocator(integer=True))
        #axs[i].set_ylim([-1.1,1.1])
        #axs[j].set_xlim([0,11])
        box = axs[j].get_position()
        axs[j].set_position([box.x0, box.y0 + box.height * 0.12,
                            box.width, box.height * 0.8])
    # for ax in axs:
    #     ax.grid(True)
    #     ax.set_xlabel("t",fontsize=12)
    #     ax.set_ylabel("$x_2$",fontsize=12)
    #     ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    #     #axs[i].set_ylim([-1.1,1.1])
    #     ax.set_xlim([0,11])
    #     box = ax.get_position()
    #     ax.set_position([box.x0, box.y0 + box.height * 0.15,
    #                         box.width, box.height * 0.8])

    for i in range(1):
        if i == 0:
            axs[0].plot(t_test,x2,label="$x_2$ Model Prediction", color='tab:blue')
            axs[0].scatter(t_test,x2_test,label="$x_2$ Test Data",color='r',marker='x')
            axs[0].set_title("ReLU", fontsize=subtitle_fontsize)

        if i == 1:
            axs[1].plot(t_test,x2_I, color='tab:blue')
            axs[1].scatter(t_test,x2_test,color='r',marker='x')
            axs[1].set_title("tanh", fontsize=subtitle_fontsize)
         
        elif i == 2:
            axs[2].plot(t_test,x2_II, color='tab:blue')
            axs[2].scatter(t_test,x2_test,color='r',marker='x')
            axs[2].set_title("$d=4$", fontsize=subtitle_fontsize)




    # Put a legend below current axis
    fig.legend(["$x_2$ Model Prediction", "$x_2$ Test Data"],loc='lower center', bbox_to_anchor=(0.5, 0.01),
            fancybox=True, shadow=False, ncol=2, fontsize=16)

    #fig.legend(["$x_2$ Model Prediction", "$x_2$ Test Data"], loc='lower center', ncol=1)
    plt.savefig('Coupled_PINN/results/PINN_RESULTS1.png')
    plt.show()