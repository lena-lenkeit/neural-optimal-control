import matplotlib.gridspec
import matplotlib.pyplot as plt
from plot_panels import *

overscale = 1.1
a4_inches = (8.27 * overscale, 11.67 * overscale)


def plot_fig2():
    fig = plt.figure(
        figsize=(a4_inches[0], a4_inches[0] / 3 * 2)
    )  # , layout="constrained")
    grid_spec = fig.add_gridspec(2, 3, wspace=0.5, hspace=0.4)
    # subfigs = fig.subfigures(2, 3)

    # Fibrosis
    ax = fig.add_subplot(grid_spec[0, 0])
    plot_placeholder(ax, placeholder="Fibrosis Model")
    # plot_panel_ref_label(ax[0, 0], ref_label="A")

    # plot_placeholder(subfigs[0, 1].gca(), placeholder="Separatrix Opt. + Const.")
    ax = fig.add_subplot(grid_spec[0, 1])
    plot_fibrosis_panel_separatrix_opt_const(ax)

    # plot_placeholder(ax[0, 2], placeholder="Curves Opt. + Const.")
    plot_fibrosis_panel_traj_opt_const(fig, grid_spec[0, 2])
    # plot_panel_ref_label(ax[0, 1], ref_label="B")

    # plot_placeholder(ax[1, 0], placeholder="Const. Phase Inset")
    plot_fibrosis_phase_grid_panel(
        fig,
        grid_spec[1, 0],
        data_filepath="paper/data/fibrosis/const_grid_scan.npz",
    )
    # plot_panel_ref_label(ax[1, 0], ref_label="C")

    # plot_placeholder(ax[1, 1], placeholder="Opt. Phase Inset")
    plot_fibrosis_phase_grid_panel(
        fig, grid_spec[1, 1], data_filepath="paper/data/fibrosis/opt_grid_scan.npz"
    )
    # plot_placeholder(ax[1, 2], placeholder="Opt. Conc. Inset")
    plot_fibrosis_conc_grid_panel(
        fig, grid_spec[1, 2], data_filepath="paper/data/fibrosis/opt_grid_scan.npz"
    )
    # plot_panel_ref_label(ax[1, 1], ref_label="D")

    plt.savefig("paper/plots/fig2_new.png", bbox_inches="tight")
    plt.savefig("paper/plots/fig2_new.svg", bbox_inches="tight")
    plt.show()


def plot_fig3():
    # Main figure
    fig = plt.figure(figsize=(a4_inches[0], a4_inches[0] / 2 * 2))
    grid_spec = fig.add_gridspec(2, 2, wspace=0.5, hspace=0.3)

    ax = fig.add_subplot(grid_spec[0, 0])
    plot_placeholder(ax, placeholder="Apoptosis Model")

    # ax = fig.add_subplot(grid_spec[0, 1])
    # plot_placeholder(ax, placeholder="Example Trajectory")
    plot_apoptosis_traj_panel(fig, grid_spec[0, 1])

    # ax = fig.add_subplot(grid_spec[1, 0])
    # plot_placeholder(ax, placeholder="Trajectory Scan")
    plot_apoptosis_traj_scan_panel(fig, grid_spec[1, 0])

    # ax = fig.add_subplot(grid_spec[1, 1])
    # plot_placeholder(ax, placeholder="Reward Scan")
    plot_apoptosis_opt_scan_panel(fig, grid_spec[1, 1])

    plt.savefig("paper/plots/fig3_new.png", bbox_inches="tight")
    plt.savefig("paper/plots/fig3_new.svg", bbox_inches="tight")
    plt.show()


def plot_fig3_extra():
    # Extra plots
    fig = plt.figure(figsize=(a4_inches[0] / 2, a4_inches[0] / 6))
    grid_spec = fig.add_gridspec(1, 2, wspace=0.5, hspace=0.0)

    ax = fig.add_subplot(grid_spec[0])
    # plot_placeholder(ax, placeholder="Receptor Activation")
    plot_apoptosis_single_event_panel(ax)

    ax = fig.add_subplot(grid_spec[1])
    # plot_placeholder(ax, placeholder="Apoptosis Single Events")
    plot_apoptosis_receptor_act_panel(ax)

    plt.savefig("paper/plots/fig3_new_extra.png", bbox_inches="tight")
    plt.savefig("paper/plots/fig3_new_extra.svg", bbox_inches="tight")
    plt.show()


def plot_fig4_extra():
    # Extra plots
    fig = plt.figure(figsize=(a4_inches[0], a4_inches[0] / 3.7))
    grid_spec = fig.add_gridspec(1, 3, wspace=0.35)

    ax = fig.add_subplot(grid_spec[0])
    plot_placeholder(ax, placeholder="Stress Model")

    ax = fig.add_subplot(grid_spec[1])
    # plot_placeholder(ax, placeholder="Prestimulation CD95L")
    plot_stress_panel_thastep_in(ax)

    ax = fig.add_subplot(grid_spec[2])
    # plot_placeholder(ax, placeholder="Prestimulation Stress")
    plot_stress_panel_thastep_out(ax)

    plt.savefig("paper/plots/fig4_new_extra.png", bbox_inches="tight")
    plt.savefig("paper/plots/fig4_new_extra.svg", bbox_inches="tight")
    plt.show()


def plot_fig5():
    # Main figure
    fig = plt.figure(figsize=(a4_inches[0], a4_inches[0] / 2 * 2))
    grid_spec = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.25)

    ax = fig.add_subplot(grid_spec[0, 0])
    plot_placeholder(ax, placeholder="Flow Schematic")

    ax = fig.add_subplot(grid_spec[0, 1])
    # plot_placeholder(ax, placeholder="Example Microscope Images")
    # plot_placeholder(ax, placeholder="Sim. Data")
    plot_stress_sim_stress_panel(ax)

    # ax = fig.add_subplot(grid_spec[1, 0])
    # plot_placeholder(ax, placeholder="Min. Peak SG")
    plot_stress_measurement_panel(fig, grid_spec[1, 0], type="minPeak")

    # ax = fig.add_subplot(grid_spec[1, 1])
    # plot_placeholder(ax, placeholder="Max. Int. SG")
    plot_stress_measurement_panel(fig, grid_spec[1, 1], type="maxMean")

    plt.savefig("paper/plots/fig5_new.png", bbox_inches="tight")
    plt.savefig("paper/plots/fig5_new.svg", bbox_inches="tight")
    plt.show()


plt.style.use("./paper/fig-style.mplstyle")
# plot_fig2()
# plot_fig3()
# plot_fig3_extra()
plot_fig4_extra()
# plot_fig5()
