from vis_utils import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Code for regenerating the plots used in the main text.
# Makes use of precomputed points. If you wish to recalculate these points see the 
# commented out calls to compute_avg_rank_points(), compute_points_ortho() and compute_points_nlihx()
# at the end of the file.

#General settings from https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Modified: dont affect height if fraction is given
    fig_height_in = (width * inches_per_pt) * golden_ratio
    #fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def setup():
    # Using seaborn's style
    plt.style.use('seaborn-ticks')
    width = 345

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble":  "".join([r'\usepackage{amssymb}', r'\usepackage{amsmath}']),
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 12,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }

    plt.rcParams.update(tex_fonts)

    colfigsize = set_size(252)
    plt.rcParams['figure.figsize'] = list(colfigsize)


def compute_points_ortho():
    """Computes averages risks after training for orthogonal training data results"""
    results_t1 = parse_process_directory("experimental_results/ortho_data/t1/")
    results_t2 = parse_process_directory("experimental_results/ortho_data/t2/")
    results_t4 = parse_process_directory("experimental_results/ortho_data/t4/")
    results_t8 = parse_process_directory("experimental_results/ortho_data/t8/")
    results_t16 = parse_process_directory("experimental_results/ortho_data/t16/")
    results_t32 = parse_process_directory("experimental_results/ortho_data/t32/")
    results_t64 = parse_process_directory("experimental_results/ortho_data/t64/")
    avgt1 = average_risk(results_t1)
    avgt2 = average_risk(results_t2)
    avgt4 = average_risk(results_t4)
    avgt8 = average_risk(results_t8)
    avgt16 = average_risk(results_t16)
    avgt32 = average_risk(results_t32)
    avgt64 = average_risk(results_t64)

    np.save("orthogonal_exp_points.npy", np.array([avgt1, avgt2, avgt4, avgt8, avgt16, avgt32, avgt64]))

def compute_points_nlihx():
    """Computes average risks after training for linearly dependent in H_X results"""
    results_t1 = parse_process_directory("experimental_results/nlihx_data/t1/")
    results_t2 = parse_process_directory("experimental_results/nlihx_data/t2/")
    results_t4 = parse_process_directory("experimental_results/nlihx_data/t4/")
    results_t8 = parse_process_directory("experimental_results/nlihx_data/t8/")
    results_t16 = parse_process_directory("experimental_results/nlihx_data/t16/")
    results_t32 = parse_process_directory("experimental_results/nlihx_data/t32/")
    results_t64 = parse_process_directory("experimental_results/nlihx_data/t64/")
    avgt1 = average_risk(results_t1)
    avgt2 = average_risk(results_t2)
    avgt4 = average_risk(results_t4)
    avgt8 = average_risk(results_t8)
    avgt16 = average_risk(results_t16)
    avgt32 = average_risk(results_t32)
    avgt64 = average_risk(results_t64)

    np.save("nlihx_exp_points.npy", np.array([avgt1, avgt2, avgt4, avgt8, avgt16, avgt32, avgt64]))


# Linearly dependent training data
def nlihx_plots():
    """Plot for linearly dependent training data. Makes use of nlihx_exp_points.npy"""
    x = np.linspace(1,64,100)
    d = 64
    expectedfn = 1 - ((d/x)**2 + d + 1)/(d*(d+1))

    # colors
    cmap = matplotlib.cm.get_cmap('tab10')
    redc = cmap(3)
    bluec = cmap(0)

    colfigsize = set_size(446) 
    fig, ax = plt.subplots(1,1,figsize=colfigsize)
    avg_risks = np.load("nlihx_exp_points.npy")
    fig.subplots_adjust(bottom=0.28)
    
    # Risk after training for trianing pairs with r such that r*t = d -> they should all have 0 risk
    ax.set_ylabel("Average risk")
    ax.set_xlabel("Number of training pairs")
    plt.axhline(y = 0, color = redc, linestyle = '--', label=r'Lower bound for the expected risk')
    plt.plot(x, expectedfn, label=r'Lower bound for linearly dependent data', color=bluec)
    xvals = [1,2,4,8,16,32,64] 
    xticks = [1,16,32,48,64]
    plt.xticks(xticks)
    plt.grid()
    plt.plot(xvals, avg_risks, 'o', label="Simulated average risks", markersize=5)
    ax.legend(bbox_to_anchor=(0, 1), loc='lower left')
    plt.ylim([-0.1,1.05])

    plt.savefig("nlihx_experiments.pdf", bbox_inches='tight')


# Orthogonal training data
def ortho_plots():
    """Plot for orthogonal training data. Makes use of orthogonal_exp_points.npy"""
    x = np.linspace(1,64,100)
    d = 64
    r = d/x
    expectedfn = 1 - (x * r**2 + d + 1)/(d*(d+1))

    # colors
    cmap = matplotlib.cm.get_cmap('tab10')
    redc = cmap(3)
    bluec = cmap(0)

    colfigsize = set_size(446) 
    fig, ax = plt.subplots(1,1,figsize=colfigsize)
    avg_risks = np.load("orthogonal_exp_points.npy")
    fig.subplots_adjust(bottom=0.28)
    
    # Risk after training for trianing pairs with r such that r*t = d -> they should all have 0 risk
    ax.set_ylabel("Average risk")
    ax.set_xlabel("Number of training pairs")
    plt.axhline(y = 0, color = redc, linestyle = '--', label=r'Lower bound for the expected risk')
    plt.plot(x, expectedfn, label=r'Lower bound for orthogonal data', color=bluec)
    xvals = [1,2,4,8,16,32,64] 
    xticks = [1,16,32,48,64]
    plt.xticks(xticks)
    plt.grid()
    plt.plot(xvals, avg_risks, 'o', label="Simulated average risks", markersize=5)
    ax.legend(bbox_to_anchor=(0, 1), loc='lower left')
    plt.ylim([-0.1,1.05])

    plt.savefig("orthogonal_experiments.pdf", bbox_inches='tight')

def compute_phases(run):
    """Computes phases after training with orthogonal data.
    Computes Uoutput = U|x> and Voutput = V|x> for each |x> in the training data (here X).
    Uoutput and Voutput should only differ by the phase of the vector which is extracted by a division.
    """
    # the phase is obtained by comparing the output of V to the output U
    Uoutput = run['U'] @ run['X']
    Voutput = run['V'] @ run['X']
    phases = Voutput / Uoutput # they should not exactly match for each sample but very closesly since loss almost 0
    return [phases[i,0,0] for i in range(0, len(phases))]

def phase_plot():
    """Plots phases for orthogonal training data on the unit circle."""
    dir = 'ortho_phases/'

    low_risk_run = { # loads a run with risk 0.3879
        'U': np.load(dir + 'low_risk_U.npy'),
        'V': np.load(dir + 'low_risk_V.npy'),
        'X': np.load(dir + 'low_risk_X.npy')
    }
    high_risk_run = { # loads a run with risk 0.8753
        'U': np.load(dir + 'high_risk_U.npy'),
        'V': np.load(dir + 'high_risk_V.npy'),
        'X': np.load(dir + 'high_risk_X.npy')
    }

    low_phases = compute_phases(low_risk_run)
    high_phases = compute_phases(high_risk_run)

    colfigsize = set_size(446) 
    print("Size: ", colfigsize)

    def create_phase_plot(both_phases, fname):
        # Plot the phases on the unit circle
        fig, axs = plt.subplots(1,2,figsize=colfigsize, sharex=True, sharey=True)
        fig.subplots_adjust(bottom=0.28)
        # gets two ax 
        for i, ax in enumerate(axs):
            ax.set_aspect('equal', 'box')

            phases = both_phases[i]

            # lines at 0
            ax.hlines(0, -2, 2, color = 'lightgrey', linewidth=0.8, zorder=0)
            ax.vlines(0, -2, 2, color = 'lightgrey', linewidth=0.8, zorder=0)

            # unit circle
            circ = plt.Circle((0, 0), radius=1.00, edgecolor='grey', facecolor='None', zorder=2, linewidth=2)
            ax.add_patch(circ)

            # Phases
            for phase in phases:
                # rescale phase a little bit since the pyplot arrows point outside of the unit circle otherwise
                ph = phase/1.05
                ax.arrow(0,0, np.real(ph), np.imag(ph), head_width=0.15, length_includes_head=True, head_length=0.15, linewidth=1, zorder=3, color='black')

            ax.set_ylim(-1.5,1.5)
            ax.set_xlim(-1.5,1.5)
            ax.set_xticks([-1,0,1])

            if i == 0:
                # ticks only for left plot
                ax.set_yticks([-1,0,1])
                ax.set_xlabel("(a)")
            else:
                ax.tick_params(axis="y", labelleft=False, length=0)
                ax.set_xlabel("(b)")
        
        # remove space
        plt.subplots_adjust(wspace=0, hspace=0)
        # overall frame
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

        plt.xlabel(r'$\mathrm{Re}(e^{i \theta_j})$')
        plt.ylabel(r'$\mathrm{Im}(e^{i \theta_j})$', labelpad=-5)

        plt.savefig(fname, bbox_inches='tight')
        return fig

    create_phase_plot([low_phases, high_phases], "ortho_phases.pdf")

def compute_avg_rank_points():
    """Computes the plot-markers for training with varying Schmidt rank data.
    Extracts average risk after training and average loss at the end of training for 
    each average rank r and each number of training samples t."""
    datadir = 'experimental_results/avg_rank_data/'
    ts = [1,2,4,8,16,32,64]
    rs = [1,2,4,64]

    num_per_point = 200

    results_full = dict()
    results_avg_risk = dict()
    results_avg_loss = dict()

    for r in rs:
        results_current = dict()
        avg_risk_current = dict()
        avg_loss_current = dict()
        for t in ts:
            results = parse_process_directory(datadir + '/t' + str(t) + 'r' + str(r) + '/')
            if len(results) < 1:
                continue # exp not finished yet
                print("Results not finished at ", (t,r))
            if len(results) < num_per_point:
                print("Results not finished - not reliable at ", (t,r))
            results_current[t] = results
            avg_risk_current[t] = average_risk(results)
            avg_loss_current[t] = np.mean(get_final_losses(results))
        results_full[r] = results_current
        results_avg_risk[r] = avg_risk_current
        results_avg_loss[r] = avg_loss_current

    np.save("avg_rank_risks.npy", results_avg_risk)
    np.save("avg_rank_losses.npy", results_avg_loss)

def avg_rank_plot():
    """Plots average risks after training for training QNNs with data of varying 
    Schmidt rank. Makes use of avg_rank_risks.npy."""
    # Plot for experiment 1 average ranks
    colfigsize = set_size(446)
    fig, ax = plt.subplots(1,1,figsize=colfigsize)

    cmap = matplotlib.cm.get_cmap('tab10')
    colors = [cmap(i) for i in range(0,4)]
    markers = ['o', "v", "*", "^"]
    lstyles = ['solid', 'dotted', 'dashed', 'dashdot']

    avg_risk = np.load("avg_rank_risks.npy", allow_pickle=True)
    avg_risk = avg_risk.item()

    dim=2**6
    rs = [64,4,2,1]

    for i, r in enumerate(rs):
        texp = np.linspace(0,dim,1000)
        expected = 1 - ((r*texp)**2 + dim + 1)/(dim*(dim+1))
        expected[expected < 0] = 0
        plt.plot(texp, expected, label="Lower bound for $\\overline{r}=" + str(r) + "$", color=colors[i], linewidth=1, linestyle=lstyles[i])

    for i, r in enumerate(rs):
        x = avg_risk[r].keys()
        y = avg_risk[r].values()

        plt.plot(x, y, label="$\\overline{r}=" + str(r) + "$", color=colors[i], marker=markers[i], markersize=5, linestyle = 'None')
    


    plt.ylabel("Average risk")
    plt.xlabel("Number of training pairs")
    plt.legend(bbox_to_anchor=(0, 1), loc='lower left', ncol=2)
    plt.xlim([0,65])
    plt.ylim([-0.05,1.05])
    plt.xticks([1,16,32,48,64])
    plt.grid()

    plt.savefig("avg_rank_experiments.pdf", bbox_inches='tight')

def avg_rank_loss_plot():
    """Plots average losses after training for training QNNs with data of varying 
    Schmidt rank. Makes use of avg_rank_losses.npy."""
    # Plot for experiment 1 average ranks
    colfigsize = set_size(446) 
    fig, ax = plt.subplots(1,1,figsize=colfigsize)

    fig.subplots_adjust(bottom=0.28)

    cmap = matplotlib.cm.get_cmap('tab10')
    colors = [cmap(i) for i in range(0,4)]
    markers = ['o', "v", "*", "^"]
    lstyles = ['solid', 'dotted', 'dashed', 'dashdot']

    avg_loss = np.load("avg_rank_losses.npy", allow_pickle=True)
    avg_loss = avg_loss.item()

    dim=2**6
    rs = [64,4,2,1]
    for i, r in enumerate(rs):
        x = avg_loss[r].keys()
        y = avg_loss[r].values()

        plt.plot(x, y, label="$\\overline{r}=" + str(r) + "$", color=colors[i], marker=markers[i], markersize=5, linestyle=lstyles[i], linewidth=1)

    ax.ticklabel_format(axis='y', style='sci', scilimits = (0,0))
    plt.ylabel("Average loss")
    plt.xlabel("Number of training pairs")
    plt.legend(bbox_to_anchor=(0.15, 1), loc='lower left', ncol=2)
    plt.xlim([0,65])
    plt.xticks([1,16,32,48,64])
    plt.grid()

    plt.savefig("avg_rank_losses.pdf", bbox_inches='tight')

if __name__ == '__main__':
    setup()
    #compute_avg_rank_points()
    #compute_points_ortho()
    #compute_points_nlihx()
    ortho_plots()
    nlihx_plots()
    phase_plot()
    avg_rank_plot()
    avg_rank_loss_plot()