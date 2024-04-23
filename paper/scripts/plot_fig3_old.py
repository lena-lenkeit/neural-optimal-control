import matplotlib.axes
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from adaptive import Learner2D
from resize_right import interp_methods, resize
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.optimize import root_scalar

a4_inches = (8.3, 11.7)


def plot_separatrix(ax: matplotlib.axes.Axes):
    separatrix_array = loadmat("paper/data/common/Separatrix_array_F06_M07.mat")

    separatrix_x = np.logspace(
        separatrix_array["lims_F"][0, 0],
        separatrix_array["lims_F"][0, 1],
        separatrix_array["tsteps"][0, 0],
    )

    separatrix_y = np.logspace(
        separatrix_array["lims_M"][0, 0],
        separatrix_array["lims_M"][0, 1],
        separatrix_array["tsteps"][0, 0],
    )

    ax.set_xlabel("Fibroblasts [Cells / ml]")
    ax.set_ylabel("Macrophages [Cells / ml]")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(1e0, 1e7)
    ax.set_ylim(1e0, 1e7)

    mesh = separatrix_array["S"].astype(np.float32)
    mesh = gaussian_filter(mesh, 5.0)

    ax.contourf(
        separatrix_x,
        separatrix_y,
        mesh,
        colors=["lightgrey", "white", "white"],
        levels=[0.0, 0.5, 1.0],
    )

    ax.contour(
        separatrix_x,
        separatrix_y,
        mesh,
        colors="black",
        levels=[0.5],
    )

    # Vector field
    def eval_vector_field_at(F, M):
        lambda1 = 0.9
        lambda2 = 0.8
        mu1 = 0.3
        mu2 = 0.3
        K = 1e6
        gamma = 2
        beta3 = 240 * 1440
        beta1 = 470 * 1440
        beta2 = 70 * 1440
        alpha1 = 940 * 1440
        alpha2 = 510 * 1440
        k1 = 6 * 1e8
        k2 = 6 * 1e8

        C = -0.5 * (alpha1 / gamma * M + k2 - beta1 / gamma * F) + np.sqrt(
            0.25 * (alpha1 / gamma * M + k2 - beta1 / gamma * F) ** 2
            + beta1 * k2 / gamma * F
        )
        P = 0.5 * (beta2 / gamma * M + (beta3 - alpha2) / gamma * F - k1) + np.sqrt(
            0.25 * (k1 - beta2 / gamma * M - (beta3 - alpha2) / gamma * F) ** 2
            + (beta2 * M + beta3 * F) * k1 / gamma
        )

        k = {}

        k[0] = 0.9  # proliferation rates: lambda1=0.9/day,
        k[1] = 0.8  # lambda2=0.8/day
        k[2] = 0.3  # mu_1, mu_2, death rates: 0.3/day
        k[3] = 1e6  # carrying capacity: 10^6 cells
        k[10] = 6e8  # #binding affinities: k1=6x10^8 molecules (PDGF)     ---- k_1
        k[11] = 6e8  # k2=6x10^8 (CSF)                                   ---- k_2
        k[12] = 0

        y = [F, M, C, P]

        dFdt = y[0] * (
            k[0] * y[3] / (k[10] + y[3]) * (1 - y[0] / k[3]) - k[2]
        )  # Fibrobasts
        dMdt = y[1] * (k[1] * y[2] / (k[11] + y[2]) - k[2]) + k[12]  # Macrophages

        return dFdt, dMdt

    F = np.geomspace(1e0, 1e7, 8)
    M = np.geomspace(1e0, 1e7, 8)
    F, M = np.meshgrid(F, M)
    dFdt, dMdt = eval_vector_field_at(F, M)
    dYdt = np.stack((dFdt, dMdt), axis=-1)
    dYdt = dYdt / np.linalg.norm(dYdt, ord=2, axis=-1, keepdims=True)

    ax.quiver(F, M, dYdt[..., 0], dYdt[..., 1], angles="xy")

    # Healing Stable Steady State
    ax.plot(
        [1],
        [1],
        marker="o",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=10,
        clip_on=False,
        zorder=100,
    )
    # Separatrix Repulsive State
    ax.plot(
        [5471],
        [1],
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=10,
        clip_on=False,
        zorder=100,
    )
    ax.plot(
        [3488],
        [957],
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=10,
        clip_on=False,
        zorder=100,
    )
    # Hot Fibrosis Stable Steady State
    ax.plot(
        [484442.90942677],
        [644403.31198653],
        marker="o",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=10,
        clip_on=False,
        zorder=100,
    )

    # Cold Fibrosis Unstable Steady State
    num_fibroblasts = root_scalar(lambda x: eval_vector_field_at(x, 0)[0], x0=5e5)
    ax.plot(
        [num_fibroblasts.root],
        [1],
        marker="o",
        markerfacecolor="white",
        markerfacecoloralt="black",
        markeredgecolor="black",
        markersize=10,
        fillstyle="top",
        clip_on=False,
        zorder=100,
    )

    ax.set_prop_cycle(None)


def plot_trajectory(
    ax: matplotlib.axes.Axes, xy: np.ndarray, label: str = None, skip: int = 4
):
    ax.plot(xy[:, 0], xy[:, 1], label=label, zorder=10)
    # ax.scatter(xy[::skip, 0], xy[::skip, 1], s=64, marker="x", zorder=10)


def plot_controls(ax: matplotlib.axes.Axes, cs: np.ndarray, const: float):
    ax.set_xlabel("Time [d]")
    ax.set_ylabel("Amount [a.u.]")

    ax.plot(cs)
    ax.plot([0, 200], [const, const], c="tab:red")

    ax.plot([], [], c="tab:blue", label=r"$\mathrm{\alpha}$PDGF")
    ax.plot([], [], c="tab:orange", label=r"$\mathrm{\alpha}$CSF-1")
    ax.plot(
        [],
        [],
        c="tab:red",
        label=r"Const. ($\mathrm{\alpha}$PDGF, $\mathrm{\alpha}$CSF-1)",
    )


def plot_fibrosis_panel_2(ax: matplotlib.axes.Axes):
    constant_sol_cs = np.load("paper/data/fibrosis/constant_sol_cs.npz")
    optimized_sol_cs = np.load("paper/data/fibrosis/optimized_sol_cs.npz")

    plot_separatrix(ax)
    plot_trajectory(ax, constant_sol_cs["solution_ys"][:, [0, 1]], label="Constant")
    plot_trajectory(ax, optimized_sol_cs["solution_ys"][:, [0, 1]], label="Optimized")
    ax.legend()


def plot_fibrosis_panel_3(ax: matplotlib.axes.Axes):
    constant_sol_cs = np.load("paper/data/fibrosis/constant_sol_cs.npz")
    optimized_sol_cs = np.load("paper/data/fibrosis/optimized_sol_cs.npz")

    plot_controls(
        ax,
        optimized_sol_cs["controller_cs"],
        constant_sol_cs["controller_cs"][0, 0],
    )
    ax.legend()


def plot_placeholder(ax: matplotlib.axes.Axes, placeholder: str = "Placeholder"):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5,
        0.5,
        s=placeholder,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )


def plot_fibrosis_panel_4(ax: matplotlib.axes.Axes):
    # Load data
    constant_learner = Learner2D(lambda x: 0, bounds=[(-6, 0), (-3, 3)])
    optimal_learner = Learner2D(lambda x: 0, bounds=[(-6, 0), (-3, 3)])

    constant_learner.load("paper/data/fibrosis/constant_learner.pickle")
    optimal_learner.load("paper/data/fibrosis/optimized_learner.pickle")

    # Prepare grid
    pdgf, csf, constant_reward = constant_learner.interpolated_on_grid()
    pdgf, csf, optimal_reward = optimal_learner.interpolated_on_grid()

    difference_reward = optimal_reward - resize(
        constant_reward,
        out_shape=optimal_reward.shape,
        interp_method=interp_methods.linear,
        pad_mode="edge",
    )

    # Plot
    ax.imshow(
        difference_reward.T,
        cmap="magma",
        extent=(pdgf[0], pdgf[-1], csf[0], csf[-1]),
        origin="lower",
        aspect="auto",
    )

    ax.set_xlim([-3, 0])
    ax.set_ylim([-3, 1])

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: "$10^{%d}$" % round(x, 1))
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: "$10^{%d}$" % round(x, 1))
    )

    ax.set_xlabel("anti-PDGF [a.u.]")
    ax.set_ylabel("anti-CSF [a.u.]")
    # ax.colorbar(fraction=0.04575, pad=0.04, label="Advantage")


def plot_apoptosis_panel_2(ax: matplotlib.axes.Axes):
    trajectory = np.load("paper/data/apoptosis/single_trajectory.npz")

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Frac. tBid")

    fractions = []
    for i, (frac, thresh) in enumerate(
        zip(trajectory["tBID_frac"].T[:3], trajectory["tBID_thresh"][:3])
    ):
        idx = np.argmax(frac >= thresh)
        fractions.append(frac[idx])

        if idx == 0:
            ax.plot(trajectory["ts"], frac, zorder=i * 2)
        else:
            (line,) = ax.plot(trajectory["ts"][:idx], frac[:idx], zorder=i * 2)
            ax.scatter(
                trajectory["ts"][idx],
                frac[idx],
                c=line.get_color(),
                marker="x",
                zorder=i * 2 + 1,
            )

            ax.axhline(frac[idx], c=line.get_color(), linestyle="dashed", zorder=-1)
            ax.annotate(
                "Apoptosis",
                (trajectory["ts"][idx], frac[idx]),
                (trajectory["ts"][idx] - 50, frac[idx] - 0.1),
                bbox=dict(boxstyle="round", fc="white", alpha=0.5),
                zorder=100,
            )

    ax.set_yticks(fractions)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: r"$\mathrm{f_{th,%s}}$" % pos)
    )


def plot_apoptosis_panel_3(ax: matplotlib.axes.Axes):
    scan_const = np.load("paper/data/apoptosis/scan_const.npy")
    scan_opt = np.load("paper/data/apoptosis/scan_opt.npy")

    ax.set_xlabel(r"CD95L Integral [$\mathrm{min {\cdot} ng / ml}$]")
    ax.set_ylabel("Frac. Dead Cells")

    ax.set_xscale("log")
    ax.plot(scan_const[:, 0], scan_const[:, 1] / 500, label="Constant")
    ax.plot(scan_opt[:, 0], scan_opt[:, 1] / 500, label="Optimized")
    ax.legend()


def plot_stress_panel_2(ax: matplotlib.axes.Axes):
    const_data = np.load("paper/data/stress/eval_data_const.npz")
    min_peak_data = np.load("paper/data/stress/eval_data_min_peak_stc.npz")
    min_mean_data = np.load("paper/data/stress/eval_data_min_mean_stc.npz")

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Frac. Stressed")

    ax.plot(const_data["ts"], const_data["sg"], label="Constant")
    (line,) = ax.plot(min_peak_data["ts"], min_peak_data["sg"], label="Min. Peak")
    ax.plot(
        min_mean_data["ts"],
        min_mean_data["sg"],
        # c=line.get_color(),
        # linestyle="dashed",
        label="Min. Int.",
    )
    ax.legend()


def plot_stress_panel_thastep_in(ax: matplotlib.axes.Axes, skip: int = 16):
    data = np.load("paper/data/stress/tha_step_scan.npz")

    cmap = plt.get_cmap("magma")
    colors = cmap(np.linspace(0, 1, 128 // skip))

    ax.set_prop_cycle(plt.cycler(color=colors))
    ax.plot(data["ts"][::skip].T, data["cs"][::skip, ..., 0].T)
    ax.set_prop_cycle(None)

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Thapsigargin [nmol/l]")


def plot_stress_panel_thastep_out(ax: matplotlib.axes.Axes, skip: int = 16):
    data = np.load("paper/data/stress/tha_step_scan.npz")

    cmap = plt.get_cmap("magma")
    colors = cmap(np.linspace(0, 1, 128 // skip))

    ax.set_prop_cycle(plt.cycler(color=colors))
    ax.plot(data["ts"][::skip].T, data["sg"][::skip].T)
    ax.set_prop_cycle(None)

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Frac. Stressed Cells")


def plot_panel_ref_label(ax: matplotlib.axes.Axes, ref_label: str):
    ax.text(
        -0.2,
        1.05,
        s=ref_label,
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
        weight="bold",
        size=20,
    )


# Plot the entire figure
plt.style.use("./paper/fig-style.mplstyle")
fig, ax = plt.subplots(5, 3, figsize=a4_inches)

# Fibrosis
plot_placeholder(ax[0, 0], placeholder="Fibrosis Model")
plot_panel_ref_label(ax[0, 0], ref_label="A1")

plot_fibrosis_panel_2(ax[0, 1])
plot_panel_ref_label(ax[0, 1], ref_label="A2")

plot_fibrosis_panel_3(ax[0, 2])
plot_panel_ref_label(ax[0, 2], ref_label="A3")

plot_fibrosis_panel_4(ax[1, 0])
plot_panel_ref_label(ax[1, 0], ref_label="A4")

fig.delaxes(ax[1, 1])
fig.delaxes(ax[1, 2])

# Apoptosis
plot_placeholder(ax[2, 0], placeholder="Apoptosis Model")
plot_panel_ref_label(ax[2, 0], ref_label="B1")

plot_apoptosis_panel_2(ax[2, 1])
plot_panel_ref_label(ax[2, 1], ref_label="B2")

plot_apoptosis_panel_3(ax[2, 2])
plot_panel_ref_label(ax[2, 2], ref_label="B3")

# Stress
plot_placeholder(ax[3, 0], placeholder="Stress Model")
plot_panel_ref_label(ax[3, 0], ref_label="C1")

plot_stress_panel_2(ax[3, 1])
plot_panel_ref_label(ax[3, 1], ref_label="C2")

fig.delaxes(ax[3, 2])

plot_stress_panel_thastep_in(ax[4, 0])
plot_panel_ref_label(ax[4, 0], ref_label="C3")

plot_stress_panel_thastep_out(ax[4, 1])
plot_panel_ref_label(ax[4, 1], ref_label="C4")

fig.delaxes(ax[4, 2])

fig.tight_layout(pad=0.75, h_pad=-0.75)
plt.savefig("paper/plots/fig3.svg", bbox_inches="tight")
plt.show()
