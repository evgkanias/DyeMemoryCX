import numpy as np

ER_pref = np.linspace(0, 4 * np.pi, 16, endpoint=False)
D7_pref = np.linspace(0, 2 * np.pi, 8, endpoint=False)

ER2EPG = -np.eye(16)

LNO2PFN = -np.kron(np.eye(2), np.ones((1, 8)))

EPG2D7 = np.tile(np.eye(8), (2, 1))

D72D7 = 0.5 * np.cos(D7_pref[:, None] - D7_pref[None, :]) - 0.5

D72PFN = -np.tile(np.eye(8), (1, 2))

D72PFL3 = -np.tile(np.eye(8), (1, 2))

FC2hD = np.eye(16)

PFN2FC2 = np.eye(16)

FC2PFL3 = np.kron(1 - np.eye(2), np.eye(8))
FC2PFL3 = np.hstack((FC2PFL3[:, 7:8], FC2PFL3[:, :7], FC2PFL3[:, 9:], FC2PFL3[:, 8:9]))

hD2PFL3 = -np.kron(np.eye(4)[::-1], np.eye(4))
hD2PFL3 = np.hstack((hD2PFL3[:, 7:8], hD2PFL3[:, :7], hD2PFL3[:, 9:], hD2PFL3[:, 8:9]))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure("model-parameters", figsize=(5, 8))
    ax = fig.subplot_mosaic("""
    AAAABB..
    AAAABB..
    AAAABB..
    AAAABB..
    AAAABBCC
    AAAABBCC
    AAAABBCC
    AAAABBCC
    DDDDFFFF
    EEEEFFFF
    GGGGFFFF
    GGGGFFFF
    GGGGHHHH
    GGGGHHHH
    GGGGHHHH
    GGGGHHHH
    GGGGHHHH
    GGGGHHHH
    JJJJHHHH
    JJJJHHHH
    JJJJLLLL
    JJJJLLLL
    NN..LLLL
    NN..LLLL
    NN..LLLL
    NN..LLLL
    NN..LLLL
    NN..LLLL
    """)

    ax["A"].imshow(ER2EPG, cmap='viridis', vmin=-1, vmax=1, origin="lower")
    ax["B"].imshow(EPG2D7, cmap='viridis', vmin=-1, vmax=1, origin="lower")
    ax["C"].imshow(D72D7, cmap='viridis', vmin=-1, vmax=1, origin="lower")
    ax["D"].imshow(LNO2PFN, cmap='viridis', vmin=-1, vmax=1, origin="lower")
    ax["E"].imshow(LNO2PFN, cmap='viridis', vmin=-1, vmax=1, origin="lower")
    ax["F"].imshow(D72PFN, cmap='viridis', vmin=-1, vmax=1, origin="lower")
    ax["G"].imshow(PFN2FC2, cmap='viridis', vmin=-1, vmax=1, origin="lower")
    ax["H"].imshow(FC2PFL3, cmap='viridis', vmin=-1, vmax=1, origin="lower")
    ax["J"].imshow(D72PFL3, cmap='viridis', vmin=-1, vmax=1, origin="lower")
    img = ax["L"].imshow(hD2PFL3, cmap='viridis', vmin=-1, vmax=1, origin="lower")
    fig.colorbar(img, ax=ax["N"])

    for ax_label, ax_i in ax.items():
        ax_i.set_xticks([])
        ax_i.set_yticks([])

    plt.tight_layout()
    plt.show()
