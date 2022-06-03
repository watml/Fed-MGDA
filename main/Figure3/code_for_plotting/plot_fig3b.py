import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    frac = 0.1
    iid = 0
    # lr, batch_size, length = 0.01, 10, 2000
    lr, batch_size, length = 0.1, 400, 3000
    seeds = [1, 5000, 6000, 9000]
    accu = np.zeros((20, 4, 1, int(length / 10))) * np.nan
    user_accu = np.zeros((20, 4, int(length / 10), 100)) * np.nan
    user_loss = np.zeros((20, 4, int(length / 10), 100)) * np.nan
    stat = []
    # folder_path = ["qffl_user/"]
    folder_path = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ]
    # res_name = [
    #     "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(0, 0, length,
    #                                                                                                        frac, iid,
    #                                                                                                        batch_size,
    #                                                                                                        lr, 1.0, 1.0,
    #                                                                                                        0.0, 0.0,
    #                                                                                                        1.0, -1, 1.0,
    #                                                                                                        0.0),
    #     "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}_vip{}_vscale{}_vbias{}".format(
    #         0, 0, length, frac, iid, batch_size, lr, 1.0, 1.0, 0.01, 0.0, 1.0, -1, 1.0, 0.0),
    #     "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(0, 1, length,
    #                                                                                                        frac, iid,
    #                                                                                                        batch_size,
    #                                                                                                        lr, 1.0, 1.0,
    #                                                                                                        0.0, 0.0,
    #                                                                                                        1.0, -1, 1.0,
    #                                                                                                        0.0),
    #     "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(1, 1, length,
    #                                                                                                        frac, iid,
    #                                                                                                        batch_size,
    #                                                                                                        lr, 1.5,
    #                                                                                                        0.89, 0.0,
    #                                                                                                        0.0, 1.0, -1,
    #                                                                                                        1.0, 0.0),
    #     "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}_vip{}_vscale{}_vbias{}".format(
    #         1, 1, length, frac, iid, batch_size, lr, 1.0, 0.89, 0.1, 0.0, 1.0, -1, 1.0, 0.0),
    #     "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(1, 0, length,
    #                                                                                                        frac, iid,
    #                                                                                                        batch_size,
    #                                                                                                        lr, 1.5,
    #                                                                                                        0.89, 0.0,
    #                                                                                                        0.0, 1.0, -1,
    #                                                                                                        1.0, 0.0),
    #     "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(0, 0, length,
    #                                                                                                        frac, iid,
    #                                                                                                        batch_size,
    #                                                                                                        lr, 1.0, 1.0,
    #                                                                                                        0.0, 0.5,
    #                                                                                                        1.0, -1, 1.0,
    #                                                                                                        0.0),
    # ]

    res_name = [
        "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(0, 0, length, frac, iid, batch_size, lr, 1.0, 1.0, 0.0, 0.0, 1.0),
        "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(0, 0, length, frac, iid, batch_size, lr, 1.0, 1.0, 0.5, 0.0, 1.0),
        "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(0, 1, length, frac, iid, batch_size, lr, 1.0, 1.0, 0.0, 0.0, 1.0),
        "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(1, 1, length, frac, iid, batch_size, lr, 1.0, 0.884, 0.0, 0.0, 1.0),
        "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(1, 1, length, frac, iid, batch_size, lr, 1.0, 0.884, 0.1, 0.0, 1.0),
        "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(1, 0, length, frac, iid, batch_size, lr, 1.0, 0.884, 0.0, 0.0, 1.0),
        "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(0, 0, length, frac, iid, batch_size, lr, 1.0, 1.0, 0.0, 0.1, 0.1),
    ]

    seed = [1, 5000, 6000, 9000]
    for idx in range(len(res_name)):
        for i, s in enumerate(seed):
            # file = folder_path[idx+1] + res_name[idx] + "_seed{}.npz".format(s)
            file = folder_path[idx] + res_name[idx] + "_seed{}.npz".format(s)
            if os.path.isfile(file):
                temp = np.load(file)
                # accu[idx, i] = temp["acc"]
                user_accu[idx, i] = temp["user"][0]
                user_loss[idx, i] = temp["user"][1]
                print("hi{}".format(idx))

    temp = []
    data = []
    for idx in range(len(res_name)):
        y = np.nanmean(accu[idx], axis=0)
        print(y[0][-1])
        cube = 100 * user_accu[idx, :, -1, :]
        data.append(cube.reshape(-1))
        # cube = np.log(user_loss[idx, :, -1, :])
        std = np.std(cube, axis=1)
        mean = np.mean(cube, axis=1)
        lower, upper = np.percentile(cube, axis=1, q=[5, 95])
        temp.append([np.mean(mean), np.std(mean), np.mean(std), np.std(std), np.mean(lower), np.std(lower),
                     np.mean(upper), np.std(upper)])
    for i in range(len(temp)):
        print("${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$".format(*temp[i]))

    label = ["FedAvg", "FedProx", "FedMGDA", "FedMGDA+", "MGDA-Prox", "FedAvg-n", r"$q$-FedAvg"]
    fig, ax = plt.subplots()

    x = np.arange(len(res_name))
    width = 0.6
    ax.set_xticks([])
    # ax.set_xticklabels(label, {'rotation': 60})
    p = ax.violinplot(data, positions=x, showmeans=True, widths=width)
    ax.set_ylabel('User test accuracy %', fontsize=45)
    # ax.grid(True)
    ax.tick_params(labelsize=45)

    val = ["cmeans", "cmaxes", "cmins", "cbars"]
    for i in val:
        p[i].set_color("black")
        p[i].set_linewidths(3)
        p[i].set_sketch_params(length=10, scale=10)
    p["cbars"].set_color("black")

    for i, pc in enumerate(p['bodies']):
        pc.set_facecolor("lightgray")
        pc.set_edgecolor('black')
        pc.set_alpha(0.2)
    # height = [10]*10
    # for i in range(len(x)):
    #     # height = rect.get_height()
    #     val = np.std(data[i])
    #     ax.annotate(r'$\sigma$={:.2f}'.format(val),
    #                 xy=(x[i], -15),
    #                 xytext=(30, 0),  # 3 points vertical offset
    #                 textcoords="offset points",
    #                 ha='center', va='bottom', fontsize=35, rotation=60)

    cmap = plt.get_cmap('hsv')
    color_map = cmap(np.linspace(0, 1.0, 7))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    print(colors)
    print(label)
    for i in range(len(data)):
        temp1 = x[i] + 0.15 * (np.random.rand(400) - 0.5)
        val = np.std(data[i])
        label_new = "$\sigma$={:.2f}".format(temp[i][2])
        plt.plot(temp1, data[i], ".", markersize=10, label=label_new, color=colors[i])
    ax.set_ylim((-19, 100))

    # ax.tick_params(direction='in')
    plt.legend(fontsize=34, loc=3, markerscale=5, ncol=4)
    # plt.legend(fontsize=45, ncol=2, markerscale=5, bbox_to_anchor=(0.048, -0.28), loc="lower left",   bbox_transform=fig.transFigure, frameon=False)
    # plt.show()
    fig = plt.gcf()

    fig.set_size_inches(22, 22 / 2)
    name = "violin_iid{}_frac{}_B{}".format(iid, frac, batch_size)
    # name = "violin"
    name = name.replace(".", "")
    fig.savefig(name + ".eps", format='eps', dpi=600, bbox_inches='tight')
    fig.savefig(name + '.png', format='png', dpi=600, bbox_inches='tight')



