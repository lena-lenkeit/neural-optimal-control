from typing import Literal, Tuple

import matplotlib.axes
import matplotlib.figure
import matplotlib.gridspec
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from adaptive import Learner2D
from resize_right import interp_methods, resize
from scipy.interpolate import interpn
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.optimize import root_scalar
from scipy.stats import bootstrap


def plot_separatrix(ax: matplotlib.axes.Axes, simple: bool = False):
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

    if simple:
        return

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

    marker_size = 7
    # Healing Stable Steady State
    ax.plot(
        [1],
        [1],
        marker="o",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=marker_size,
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
        markersize=marker_size,
        clip_on=False,
        zorder=100,
    )
    ax.plot(
        [3488],
        [957],
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=marker_size,
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
        markersize=marker_size,
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
        markersize=marker_size,
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


def plot_fibrosis_controls(ax: matplotlib.axes.Axes, cs: np.ndarray, num_days: int):
    ax.set_xlabel("Time [d]")
    ax.set_ylabel("Amount [a.u.]")

    ax.plot(cs[:num_days])

    ax.plot([], [], c="tab:blue", label=r"$\mathrm{\alpha}$PDGF")
    ax.plot([], [], c="tab:orange", label=r"$\mathrm{\alpha}$CSF-1")


def plot_fibrosis_amounts(ax: matplotlib.axes.Axes, ys: np.ndarray, num_days: int):
    ax.set_xlabel("Time [d]")
    ax.set_ylabel("Num. Cells\n[Cells / ml]")

    ax.plot(ys[:num_days])

    ax.set_yscale("log")
    ax.plot([], [], c="tab:blue", label=r"Fibroblasts")
    ax.plot([], [], c="tab:orange", label=r"Macrophages")


def plot_fibrosis_panel_separatrix_opt_const(ax: matplotlib.axes.Axes):
    constant_sol_cs = np.load("paper/data/fibrosis/constant_sol_cs.npz")
    optimized_sol_cs = np.load("paper/data/fibrosis/optimized_sol_cs.npz")

    plot_separatrix(ax)
    plot_trajectory(ax, constant_sol_cs["solution_ys"][:, [0, 1]], label="Constant")
    plot_trajectory(ax, optimized_sol_cs["solution_ys"][:, [0, 1]], label="Optimized")
    ax.legend()


def plot_fibrosis_panel_traj_opt_const(
    fig: matplotlib.figure.Figure, subplot_spec: matplotlib.gridspec.SubplotSpec
):
    constant_sol_cs = np.load("paper/data/fibrosis/constant_sol_cs.npz")
    optimized_sol_cs = np.load("paper/data/fibrosis/optimized_sol_cs.npz")

    grid_spec = subplot_spec.subgridspec(2, 2)

    ax_const_cs = fig.add_subplot(grid_spec[0, 0])
    plot_fibrosis_controls(ax_const_cs, constant_sol_cs["controller_cs"], num_days=30)
    ax_const_cs.legend()
    ax_const_cs.tick_params(labelbottom=False)
    ax_const_cs.set_xlabel("")
    ax_const_cs.set_title("Constant")

    ax_opt_cs = fig.add_subplot(grid_spec[0, 1], sharex=ax_const_cs, sharey=ax_const_cs)
    plot_fibrosis_controls(ax_opt_cs, optimized_sol_cs["controller_cs"], num_days=30)
    ax_opt_cs.tick_params(labelbottom=False)
    ax_opt_cs.set_xlabel("")
    ax_opt_cs.tick_params(labelleft=False)
    ax_opt_cs.set_ylabel("")
    ax_opt_cs.set_title("Optimized")

    ax_const_ys = fig.add_subplot(grid_spec[1, 0], sharex=ax_const_cs)
    plot_fibrosis_amounts(
        ax_const_ys, constant_sol_cs["solution_ys"][:, :2], num_days=30
    )
    ax_const_ys.legend()

    ax_opt_ys = fig.add_subplot(grid_spec[1, 1], sharex=ax_const_cs, sharey=ax_const_ys)
    plot_fibrosis_amounts(
        ax_opt_ys, optimized_sol_cs["solution_ys"][:, :2], num_days=30
    )
    ax_opt_ys.tick_params(labelleft=False)
    ax_opt_ys.set_ylabel("")


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


def plot_apoptosis_single_event_panel(ax: matplotlib.axes.Axes):
    trajectory = np.load("paper/data/apoptosis/single_trajectory.npz")

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("tBID")

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
                (-2, -5),
                va="top",
                ha="right",
                xycoords="data",
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="white", alpha=0.5),
                size=8,
                zorder=100,
            )

    ax.set_yticks(fractions)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: r"$\mathrm{f_{th,%s}}$" % pos)
    )


def plot_apoptosis_traj_scan_panel(
    fig: matplotlib.figure.Figure, subplot_spec: matplotlib.gridspec.SubplotSpec
):
    reaction_volume = 0.8  # ml
    concentration_to_moles = 16.6 / 500  # 16.6nM = 500ng/ml
    conversion_factor = reaction_volume / concentration_to_moles

    scan_opt_controls = np.load("paper/data/apoptosis/scan_opt_controls.npz")
    single_full = np.load("paper/data/apoptosis/single_full.npz")

    single_mean_conc = single_full["mean_cd95l_conc"]
    mean_concs = scan_opt_controls["mean_cd95l_concs"]
    curves = scan_opt_controls["cd95l_control_curves"]
    times = scan_opt_controls["cd95l_control_times"]

    curves = curves[..., 0]

    # Crop to ROI
    curves = curves[10:-7]
    mean_concs = mean_concs[10:-7]

    # Interpolate from unevenly-spaced CD95L levels to evenly-spaced levels
    eval_x = times
    eval_y = np.linspace(np.log(mean_concs[0]), np.log(mean_concs[-1]), len(times))
    values = interpn(
        (times, np.log(mean_concs)),
        curves.T,
        np.stack(np.meshgrid(eval_x, eval_y), axis=-1),
    )

    # Plot trajectories
    grid_spec = subplot_spec.subgridspec(1, 2, width_ratios=[1.0, 0.1])
    ax = fig.add_subplot(grid_spec[0])
    imshow = ax.imshow(
        values,
        cmap="magma",
        norm="log",
        aspect="auto",
        origin="lower",
        extent=[
            times[0],
            times[-1],
            np.log10(mean_concs[0] * conversion_factor * 180 / 60),
            np.log10(mean_concs[-1] * conversion_factor * 180 / 60),
        ],
    )

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"$10^{int(x)}$"))

    ax.set_xlabel("Time [min]")
    ax.set_ylabel(r"CD95L Time Integral [$\mathrm{h {\cdot} ng / ml}$]")

    # Plot max-average-peak line
    avg_max_act_cd95l_conc = 2795.5549439517386
    ax.axhline(
        np.log10(avg_max_act_cd95l_conc * 180 / 60), color="tab:red", linewidth=2
    )
    ax.axhline(
        np.log10(single_mean_conc * conversion_factor * 180 / 60),
        color="xkcd:yellow",
        linewidth=2,
    )

    ax = fig.add_subplot(grid_spec[1])
    cbar = plt.colorbar(imshow, cax=ax)
    cbar.set_label("CD95L [ng/ml]")


def plot_apoptosis_traj_panel(
    fig: matplotlib.figure.Figure, subplot_spec: matplotlib.gridspec.SubplotSpec
):
    reaction_volume = 0.8  # ml
    concentration_to_moles = 16.6 / 500  # 16.6nM = 500ng/ml
    conversion_factor = reaction_volume / concentration_to_moles

    data = np.load("paper/data/apoptosis/single_full.npz")

    time = data["time"]
    mean_conc = data["mean_cd95l_conc"]
    opt_tbid_frac = data["opt_tbid_frac"]
    opt_tbid_frac_mean = data["opt_tbid_frac_mean"]
    opt_tbid_frac_std = data["opt_tbid_frac_std"]
    const_tbid_frac = data["const_tbid_frac"]
    const_tbid_frac_mean = data["const_tbid_frac_mean"]
    const_tbid_frac_std = data["const_tbid_frac_std"]
    opt_dead_cells = data["opt_dead_cells"]
    const_dead_cells = data["const_dead_cells"]
    cd95l_curve = data["cd95l_curve"]

    grid_spec = subplot_spec.subgridspec(3, 1)
    ax1 = fig.add_subplot(grid_spec[0])
    ax2 = fig.add_subplot(grid_spec[1], sharex=ax1)
    ax3 = fig.add_subplot(grid_spec[2], sharex=ax1)
    ax = [ax1, ax2, ax3]

    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)

    # tBID
    ax[0].set_ylabel("Rel. tBID")

    ax[0].set_ylim([-0.05, 1.05])
    ax[0].fill_between(
        time,
        const_tbid_frac_mean - const_tbid_frac_std,
        const_tbid_frac_mean + const_tbid_frac_std,
        color="tab:orange",
        alpha=0.25,
    )
    ax[0].plot(time, const_tbid_frac_mean, c="tab:orange")

    ax[0].fill_between(
        time,
        opt_tbid_frac_mean - opt_tbid_frac_std,
        opt_tbid_frac_mean + opt_tbid_frac_std,
        color="tab:blue",
        alpha=0.25,
    )
    ax[0].plot(time, opt_tbid_frac_mean, c="tab:blue")

    # Number of Cells
    ax[1].set_ylabel("Frac. Dead")
    ax[1].set_ylim([-0.05, 1.05])
    ax[1].plot(time, const_dead_cells / 500, c="tab:orange")
    ax[1].plot(time, opt_dead_cells / 500, c="tab:blue")

    # Control
    ax[2].set_xlabel("Time [min]")
    ax[2].set_ylabel("CD95L [ng/ml]")
    ax[2].set_yscale("log")

    optimal_receptor_activation_inst_cd95l = 2795.5549439517386
    ax[2].axhline(optimal_receptor_activation_inst_cd95l, c="tab:red", linestyle="--")

    ax[2].plot(
        [time[0], time[-1]],
        [mean_conc * conversion_factor] * 2,
        c="tab:orange",
    )

    ax[2].plot(
        time,
        cd95l_curve * conversion_factor,
        c="tab:blue",
    )

    """
    ax[2].text(
        0.95,
        0.95,
        f"Total CD95L = {mean_conc.item() * conversion_factor * 180:.1e} min*ng/ml",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax[2].transAxes,
        size=10,
        bbox=dict(boxstyle="square", facecolor="white", alpha=0.75),
    )
    """

    # ax[2].plot([], [], color="tab:blue", label="Optimized")
    # ax[2].plot([], [], color="tab:orange", label="Constant")
    # ax[2].plot([], [], linestyle="--", color="tab:red", label="Max. Recept. Act.")
    # ax[2].legend()

    ax[0].plot([], [], color="tab:blue", label="Optimized")
    ax[0].plot([], [], color="tab:orange", label="Constant")
    ax[0].legend()


def plot_apoptosis_opt_scan_panel(
    fig: matplotlib.figure.Figure, subplot_spec: matplotlib.gridspec.SubplotSpec
):
    scan_const = np.load("paper/data/apoptosis/scan_const.npz")
    scan_opt = np.load("paper/data/apoptosis/scan_opt.npz")

    grid_spec = subplot_spec.subgridspec(2, 1)

    ax_reward = fig.add_subplot(grid_spec[0])
    ax_frac = fig.add_subplot(grid_spec[1], sharex=ax_reward)
    ax_reward.tick_params(labelbottom=False)

    const_rewards = scan_const["proxy_rewards"]
    opt_rewards = scan_opt["proxy_rewards"]

    ax_reward.set_ylabel("Proxy Reward")
    ax_reward.set_xscale("log")
    ax_reward.plot(opt_rewards[:, 0] / 60, opt_rewards[:, 1] / 500, label="Optimized")
    ax_reward.plot(
        const_rewards[:, 0] / 60, const_rewards[:, 1] / 500, label="Constant"
    )
    ax_reward.legend()

    const_frac = scan_const["dead_cells"]
    opt_frac = scan_opt["dead_cells"]

    ax_frac.set_xlabel(r"CD95L Time Integral [$\mathrm{h {\cdot} ng / ml}$]")
    ax_frac.set_ylabel("Frac. Dead Cells")
    ax_frac.set_xscale("log")
    ax_frac.plot(opt_frac[:, 0] / 60, opt_frac[:, 1] / 500, label="Optimized")
    ax_frac.plot(const_frac[:, 0] / 60, const_frac[:, 1] / 500, label="Constant")


def plot_apoptosis_receptor_act_panel(ax: matplotlib.axes.Axes):
    optimal_receptor_activation_inst_cd95l = 2795.5549439517386

    receptor_data = np.load("paper/data/apoptosis/receptor_act.npz")

    cd95l_conc = receptor_data["cd95l_conc"]
    receptor_act = receptor_data["receptor_act"]

    ax.set_xscale("log")
    ax.set_xlabel("CD95L [ng/ml]")
    ax.set_ylabel("Norm. Act.")
    ax.plot(cd95l_conc, receptor_act)
    ax.axvline(optimal_receptor_activation_inst_cd95l, color="tab:red")


def plot_stress_sim_stress_panel(ax: matplotlib.axes.Axes):
    const_data = np.load("paper/data/stress/eval_data_const.npz")
    min_peak_data = np.load("paper/data/stress/eval_data_min_peak_stc.npz")
    min_mean_data = np.load("paper/data/stress/eval_data_min_mean_stc.npz")

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Frac. Stressed Cells")

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


def plot_stress_measurement_panel(
    fig: matplotlib.figure.Figure,
    subplot_spec: matplotlib.gridspec.SubplotSpec,
    type: Literal["minPeak", "maxMean"],
):
    if type == "minPeak":
        model_sg = np.load("paper/data/stress/fit/min_peak_sgs.npy")
        tha_curve = np.load("paper/data/stress/eval_data_min_peak.npz")
        exp_summary_data = [
            np.load(f"paper/data/stress/measured/minSGamp_{i}_summary_data.npz")
            for i in [230921, 230928, 231010]
        ]
    if type == "maxMean":
        model_sg = np.load("paper/data/stress/fit/max_mean_sgs.npy")
        tha_curve = np.load("paper/data/stress/eval_data_max_mean.npz")
        exp_summary_data = [
            np.load(f"paper/data/stress/measured/maxSGint_{i}_summary_data.npz")
            for i in [230923, 231003, 231028]
        ]

    # print(list(exp_summary_data[0].keys()))

    grid_spec = subplot_spec.subgridspec(2, 1)

    ax = fig.add_subplot(grid_spec[0])
    ax.tick_params(labelbottom=False)
    ax.set_ylabel(r"Thapsigargin [nmol/l]")
    ax.plot(tha_curve["ts"][:600], tha_curve["cs"][:600])
    ax.fill_between(
        tha_curve["ts"][:600],
        tha_curve["cs"][:600, 0],
        np.zeros_like(tha_curve["cs"][:600, 0]),
        alpha=0.5,
    )

    ax = fig.add_subplot(grid_spec[1], sharex=ax)
    ax.set_xlabel(r"Time [min]")
    ax.set_ylabel(r"Frac. Stressed Cells")
    ax.set_ylim([-0.05, 1.05])
    ax.plot(tha_curve["ts"][:600], model_sg[:600], c="black", linestyle="--", zorder=10)

    for exp in exp_summary_data:
        time = exp["timepoints"][:42]
        frac_cells = exp["frac_stressed_cells"][:42]
        num_stressed = exp["num_stressed_cells"][:42]
        num_cells = exp["num_nuclei"]
        var = num_cells * frac_cells * (1 - frac_cells)
        std = np.sqrt(var)

        (line,) = ax.plot(time, frac_cells)
        ax.fill_between(
            time,
            (num_stressed + std * 3) / num_cells,
            (num_stressed - std * 3) / num_cells,
            color=line.get_color(),
            alpha=0.2,
        )

    ax.plot([], [], c="black", linestyle="--", label="Model")
    ax.plot([], [], c="black", linestyle="-", label="Experiment")
    ax.legend()


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


def make_grid_plot(
    fig: matplotlib.figure.Figure, subplot_spec: matplotlib.gridspec.SubplotSpec, n: int
):
    # Make grid of plots
    grid_spec = subplot_spec.subgridspec(
        n + 1, n + 1, width_ratios=[1e-5] + [1.0] * n, height_ratios=[1.0] * n + [1e-5]
    )

    # Add rows back as single large axis
    left_ax = fig.add_subplot(grid_spec[:-1, 0])
    bottom_ax = fig.add_subplot(grid_spec[-1, 1:])

    # Remove extra axes
    left_ax.spines[["right", "top", "bottom"]].set_visible(False)
    left_ax.tick_params(axis="x", bottom=False, labelbottom=False)

    bottom_ax.spines[["right", "left", "top"]].set_visible(False)
    bottom_ax.tick_params(axis="y", left=False, labelleft=False)

    # Make main grid
    main_ax = np.empty((n, n), dtype=np.object_)
    for i in range(n):
        for j in range(n):
            main_ax[i, j] = fig.add_subplot(grid_spec[n - i - 1, j + 1])

    base_ax = main_ax[0, 0]

    for ax in main_ax.flatten():
        if ax != base_ax:
            # Share axes
            ax.sharex(base_ax)
            ax.sharey(base_ax)

        # Remove ticks
        ax.tick_params(
            axis="both", left=False, labelleft=False, bottom=False, labelbottom=False
        )

    return main_ax, left_ax, bottom_ax


def plot_fibrosis_phase_grid_panel(
    fig: matplotlib.figure.Figure,
    subplot_spec: matplotlib.gridspec.SubplotSpec,
    data_filepath: str,
    title: str = "",
):
    grid_scan = np.load(data_filepath)

    main_ax, left_ax, bottom_ax = make_grid_plot(fig, subplot_spec, n=10)

    left_ax.set_ylabel(r"$\mathrm{\alpha}$CSF-1")
    bottom_ax.set_xlabel(r"$\mathrm{\alpha}$PDGF")

    left_ax.set_yscale("log")
    left_ax.set_ylim([1e-3, 1e0])

    bottom_ax.set_xscale("log")
    bottom_ax.set_xlim([1e-3, 1e0])

    ax = main_ax.flatten()
    grid_ys = grid_scan["grid_ys"]
    grid_valid = grid_scan["grid_valid"]

    for i, _ax in enumerate(ax):
        # Handle invalid data
        if not grid_valid[i]:
            _ax.set_facecolor("red")
            continue

        # Determine outcome
        ys = grid_ys[i][:, :2]
        ys_end = ys[-1]

        threshold = 1e2
        outcome = None
        if ys_end[0] < threshold and ys_end[1] < threshold:
            outcome = "healing"
        elif ys_end[0] > threshold and ys_end[1] < threshold:
            outcome = "cold"
        elif ys_end[0] > threshold and ys_end[1] > threshold:
            outcome = "hot"
        else:
            raise ValueError()

        colors = {"healing": "tab:green", "cold": "tab:blue", "hot": "tab:red"}

        # Set plot border color based on outcome
        for spine in _ax.spines.values():
            spine.set_edgecolor(colors[outcome])
            spine.set_linewidth(1)

        # Plot data
        plot_separatrix(_ax, simple=True)
        _ax.plot(*ys.T)

        # Correct labels
        _ax.set_xlabel("")
        _ax.set_ylabel("")

    # main_ax[0, 0].set_xlabel("F")
    # main_ax[0, 0].set_ylabel("M")


def plot_fibrosis_conc_grid_panel(
    fig: matplotlib.figure.Figure,
    subplot_spec: matplotlib.gridspec.SubplotSpec,
    data_filepath: str,
):
    grid_scan = np.load(data_filepath)

    main_ax, left_ax, bottom_ax = make_grid_plot(fig, subplot_spec, n=10)

    left_ax.set_ylabel(r"$\mathrm{\alpha}$CSF-1")
    bottom_ax.set_xlabel(r"$\mathrm{\alpha}$PDGF")

    left_ax.set_yscale("log")
    left_ax.set_ylim([1e-3, 1e0])

    bottom_ax.set_xscale("log")
    bottom_ax.set_xlim([1e-3, 1e0])

    ax = main_ax.flatten()
    grid_cs = grid_scan["grid_cs"]
    grid_valid = grid_scan["grid_valid"]

    for i, _ax in enumerate(ax):
        # Handle invalid data
        if not grid_valid[i]:
            _ax.set_facecolor("red")
            continue

        cs = grid_cs[i]
        _ax.plot(cs)

    # main_ax[0, 0].set_xlabel("F")
    # main_ax[0, 0].set_ylabel("M")
