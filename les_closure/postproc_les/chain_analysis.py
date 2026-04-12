import os
import sys
import numpy as np

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib import cm

LES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if LES_ROOT not in sys.path:
    sys.path.insert(0, LES_ROOT)


single_column = 255
oneandhalf_column = 397
double_column = 539
fig_width_pt = double_column
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean       # height in inches
fig_size = [fig_width, fig_height]


mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)
# plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
# mpl.rcParams['legend.frameon'] = False
# plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1

def main(args=None):

    case_name = '3_params_both'
    n_chains = 8

    case_path = os.path.join(LES_ROOT, 'output')
    plot_path = os.path.join(case_path, 'chain_analysis_plots')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    # ===============================
    # LOAD DATA
    # ===============================
    for i in range(n_chains):
        data_path = os.path.join(case_path, f'chain{i}_0.npz')

        chain = np.load(data_path)['C']
        chain_dist = np.load(data_path)['dist']
        print(chain.keys())

    print("Loaded data from:", data_path)
    
    # adjust keys if needed
    C = data["C"]        # shape: (N_total, N_params)
    dist = data["dist"]

    # ---- infer number of chains ----
    N_total, N_params = C.shape
    N_per_chain = N_total // n_chains

    chains = C.reshape(n_chains, N_per_chain, N_params)
    dist_chains = dist.reshape(n_chains, N_per_chain)

    print(f"{n_chains} chains, {N_per_chain} samples each, {N_params} params")

    params_names = [r'$C_1$', r'$C_2$', r'$C_3$']
    C_limits = [[-0.075, -0.015],
                [-0.45,   0.45 ],
                [-0.395,  0.165]]
    # ===============================
    # POSTERIOR HISTOGRAMS
    # ===============================
    fig, ax = plt.subplots(1, N_params, figsize=(fig_width*1.5, fig_height), sharey='row')
    for p in range(N_params):
        for c in range(n_chains):
            pdf, edges = np.histogram(chains[c,:,p], bins=100, range=C_limits[p], density=True)
            x = (edges[1:] + edges[:-1]) / 2
            ax[p].plot(x, pdf, alpha=0.5)
        ax[p].set_title(f"Posterior param {p}")
        ax[p].set_xlabel(params_names[p])
        ax[p].set_ylabel("Density")
        fig.savefig(os.path.join(plot_path, f"hist_params"))

    # ===============================
    # TRACE PLOTS
    # ===============================
    fig, ax = plt.subplots(N_params, 1, figsize=(5, N_params*3), sharex='col')
    for p in range(N_params):
        for c in range(n_chains):
            ax[p].plot(chains[c, :, p], alpha=0.6)
        ax[p].set_ylabel(params_names[p])
    ax[0].set_title("Trace plots")
    ax[-1].set_xlabel("Iteration")
    fig.savefig(os.path.join(plot_path, f"trace_params"))

    # ===============================
    # DISTANCE TRACE
    # ===============================
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    for c in range(n_chains):
        ax.plot(dist_chains[c], alpha=0.5)
    ax.set_title("Distance trace")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance")
    fig.savefig(os.path.join(plot_path, f"trace_distance")) 

    # ===============================
    # AUTOCORRELATION
    # ===============================
    def autocorr(x, lag):
        return np.corrcoef(x[:-lag], x[lag:])[0,1]

    max_lag = 50

    for p in range(N_params):
        fig = plt.figure(figsize=(4, 3))
        ax = plt.gca()
        for c in range(n_chains):
            acf = [autocorr(chains[c,:,p], lag) for lag in range(1, max_lag)]
            ax.plot(acf, alpha=0.5)
        ax.set_title(f"Autocorrelation param {p}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        fig.savefig(os.path.join(plot_path, f"autocorr_param_{p}"))
    
    # ===============================
    # R-HAT (Gelman-Rubin)
    # ===============================
    def compute_rhat(chains):
        m, n, d = chains.shape
        rhat = np.zeros(d)

        for p in range(d):
            samples = chains[:, :, p]

            chain_means = np.mean(samples, axis=1)
            grand_mean = np.mean(chain_means)

            B = n * np.var(chain_means, ddof=1)
            W = np.mean(np.var(samples, axis=1, ddof=1))

            var_hat = (1 - 1/n) * W + B/n
            rhat[p] = np.sqrt(var_hat / W)

        return rhat

    rhat = compute_rhat(chains)
    print("R-hat:", rhat)

    # ===============================
    # EFFECTIVE SAMPLE SIZE (ESS)
    # ===============================
    def compute_ess(chain):
        x = chain - np.mean(chain)
        n = len(x)
        acf_sum = 0.0

        for lag in range(1, n//2):
            rho = np.corrcoef(x[:-lag], x[lag:])[0,1]
            if rho <= 0:
                break
            acf_sum += 2*rho

        return n / (1 + acf_sum)

    for p in range(N_params):
        ess_vals = [compute_ess(chains[c,:,p]) for c in range(n_chains)]
        print(f"ESS param {p}: mean={np.mean(ess_vals):.1f}, per chain={ess_vals}")

    # ===============================
    # REJECTION / STICKINESS CHECK
    # ===============================
    for c in range(n_chains):
        repeats = np.mean(np.all(chains[c,1:] == chains[c,:-1], axis=1))
        print(f"Chain {c} repeated states fraction: {repeats:.4f}")

if __name__ == "__main__":
    main()