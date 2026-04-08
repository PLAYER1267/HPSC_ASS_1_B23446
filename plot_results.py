"""
plot_results.py
===============
Publication-quality plots for the DEM Assignment 1 report.
Reads from results/ directory produced by dem_solver.

Usage:
    python3 plot_results.py [results_dir]

Produces all PDF figures required by the IEEE white paper.
"""

import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ---------- Global style ----------
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        12,
    'axes.titlesize':   13,
    'axes.labelsize':   12,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'legend.fontsize':  11,
    'figure.dpi':       150,
    'lines.linewidth':  1.8,
    'grid.alpha':       0.3,
    'axes.grid':        True,
})

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
FIGS = Path("figures"); FIGS.mkdir(exist_ok=True)

def save(fig, name):
    p = FIGS / (name + ".pdf")
    fig.tight_layout()
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {p}")

def load(path, **kwargs):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, **kwargs)

# ============================================================
#  1. Free Fall: numerical vs analytical
# ============================================================
def plot_freefall():
    df = load(ROOT / "verification/free_fall/analytical_comparison.csv")
    if df is None: print("  [skip] free_fall analytical_comparison.csv"); return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.plot(df['t'], df['z_numerical'],  lw=2,   label='Numerical')
    ax.plot(df['t'], df['z_analytical'], lw=2, ls='--', label='Analytical')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('z (m)')
    ax.set_title('Free Fall: Position'); ax.legend()

    ax = axes[1]
    ax.plot(df['t'], df['vz_numerical'],  lw=2,   label='Numerical')
    ax.plot(df['t'], df['vz_analytical'], lw=2, ls='--', label='Analytical')
    ax.set_xlabel('Time (s)'); ax.set_ylabel(r'$v_z$ (m/s)')
    ax.set_title('Free Fall: Velocity'); ax.legend()

    ax = axes[2]
    ax.semilogy(df['t'], df['error_z'], lw=2, color='firebrick')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('|Error| (m)')
    ax.set_title('Free Fall: Position Error')

    save(fig, "fig_freefall")

# ============================================================
#  2. Timestep convergence
# ============================================================
def plot_dt_error():
    df = load(ROOT / "verification/dt_sensitivity/dt_scan.csv")
    if df is None: print("  [skip] dt_scan.csv"); return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(df['dt'], df['error_z'],  'o-', label='Position error $|\\Delta z|$')
    ax.loglog(df['dt'], df['error_vz'], 's--', label='Velocity error $|\\Delta v_z|$')

    # O(dt) reference line
    dt = np.array(df['dt'])
    ref = dt / dt[-1] * float(df['error_z'].iloc[-1])
    ax.loglog(dt, ref, 'k:', lw=1.2, label=r'$O(\Delta t)$')

    ax.set_xlabel(r'Timestep $\Delta t$ (s)')
    ax.set_ylabel('Absolute error')
    ax.set_title('Timestep Convergence (Free Fall)')
    ax.legend()
    save(fig, "fig_dt_error")

# ============================================================
#  3. Bounce: height + KE vs time, two damping values
# ============================================================
def plot_bounce():
    labels = [
        ("bounce_light", "verification/bounce_light", r"$\gamma_n=20$"),
        ("bounce_heavy", "verification/bounce_heavy", r"$\gamma_n=70$"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ['steelblue', 'darkorange']

    for (key, subdir, lbl), col in zip(labels, colors):
        df = load(ROOT / subdir / "diagnostics.csv")
        if df is None: continue
        axes[0].plot(df['time'], df['min_height'], lw=1.8, color=col, label=lbl)
        axes[1].plot(df['time'], df['kinetic_energy'], lw=1.8, color=col, label=lbl)

    axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Min height (m)')
    axes[0].set_title('Bouncing Particle: Height vs Time')
    axes[0].set_ylim(bottom=0); axes[0].legend()

    axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Kinetic Energy (J)')
    axes[1].set_title('Bouncing Particle: KE vs Time'); axes[1].legend()

    save(fig, "fig_bounce")

# ============================================================
#  4. Profiling pie chart
# ============================================================
def plot_profiling():
    # Use N=1000 serial profiling result
    df = load(ROOT / "profiling/n1000_serial_pairs/profile.csv")
    if df is None:
        # try experiment directory
        df = load(ROOT / "experiments/n1000_serial_pairs/profile.csv")
    if df is None: print("  [skip] profile.csv"); return

    fig, ax = plt.subplots(figsize=(6, 5))
    colors  = ['#e15759','#4e79a7','#f28e2b','#76b7b2','#59a14f','#b07aa1']
    explode = [0.06 if f == df['percent'].max() else 0 for f in df['percent']]
    wedges, texts, autotexts = ax.pie(
        df['percent'], labels=df['function'],
        autopct='%1.1f%%', colors=colors, explode=explode, startangle=90)
    for at in autotexts: at.set_fontsize(10)
    ax.set_title('Runtime Distribution (N=1000, Serial)')
    save(fig, "fig_profiling")

# ============================================================
#  5. Strong scaling
# ============================================================
def plot_strong_scaling():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ['#1f77b4', '#ff7f0e']

    for Nval, col in zip([1000, 5000], colors):
        df = load(ROOT / f"strong_scaling_N{Nval}" / f"speedup_N{Nval}.csv")
        if df is None: continue
        axes[0].plot(df['threads'], df['speedup'],    'o-', color=col, label=f'N={Nval}')
        axes[1].plot(df['threads'], df['efficiency'], 's--', color=col, label=f'N={Nval}')

    tmax = 8
    axes[0].plot([1, tmax], [1, tmax], 'k:', lw=1.2, label='Ideal')
    axes[1].axhline(1.0, color='k', ls=':', lw=1.2, label='Ideal')

    axes[0].set_xlabel('Threads'); axes[0].set_ylabel('Speedup $S(p)$')
    axes[0].set_title('Strong Scaling: Speedup'); axes[0].legend()

    axes[1].set_xlabel('Threads'); axes[1].set_ylabel('Efficiency $E(p)$')
    axes[1].set_title('Strong Scaling: Efficiency')
    axes[1].set_ylim(0, 1.15); axes[1].legend()

    save(fig, "fig_strong_scaling")

# ============================================================
#  6. Weak scaling
# ============================================================
def plot_weak_scaling():
    df = load(ROOT / "weak_scaling/weak_scaling.csv")
    if df is None: print("  [skip] weak_scaling.csv"); return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(df['threads'], df['runtime_s'], 'o-', color='steelblue')
    axes[0].axhline(df['runtime_s'].iloc[0], color='k', ls=':', lw=1.2, label='Ideal (constant)')
    axes[0].set_xlabel('Threads'); axes[0].set_ylabel('Runtime (s)')
    axes[0].set_title('Weak Scaling: Runtime'); axes[0].legend()

    axes[1].plot(df['threads'], df['weak_efficiency'], 's-', color='darkorange')
    axes[1].axhline(1.0, color='k', ls=':', lw=1.2)
    axes[1].set_xlabel('Threads'); axes[1].set_ylabel('Weak Efficiency')
    axes[1].set_title('Weak Scaling: Efficiency')
    axes[1].set_ylim(0, 1.1)

    save(fig, "fig_weak_scaling")

# ============================================================
#  7. Neighbour search comparison
# ============================================================
def plot_neighbor():
    df = load(ROOT / "neighbor_bonus/neighbor_comparison.csv")
    if df is None: print("  [skip] neighbor_comparison.csv"); return

    Ns     = sorted(df['N'].unique())
    pairs  = df[df['mode'] == 'all_pairs'].set_index('N')
    grid   = df[df['mode'] == 'grid'].set_index('N')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(Ns)); w = 0.35
    axes[0].bar(x - w/2, [pairs.loc[n,'runtime_s']       for n in Ns], w, label='All-pairs', color='#e15759')
    axes[0].bar(x + w/2, [grid.loc[n,'runtime_s']        for n in Ns], w, label='Grid',      color='#4e79a7')
    axes[0].set_xticks(x); axes[0].set_xticklabels([f'N={n}' for n in Ns])
    axes[0].set_ylabel('Runtime (s)'); axes[0].set_title('Runtime vs N')
    axes[0].legend()

    axes[1].bar(x - w/2, [pairs.loc[n,'candidate_pairs'] for n in Ns], w, label='All-pairs', color='#e15759')
    axes[1].bar(x + w/2, [grid.loc[n,'candidate_pairs']  for n in Ns], w, label='Grid',      color='#4e79a7')
    axes[1].set_xticks(x); axes[1].set_xticklabels([f'N={n}' for n in Ns])
    axes[1].set_ylabel('Candidate pairs checked')
    axes[1].set_title('Candidate Pairs vs N'); axes[1].legend()
    axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{x/1e6:.1f}M' if x>=1e6 else f'{x:.0f}'))

    save(fig, "fig_neighbor")

# ============================================================
#  8. Kinetic energy — multi-particle settling
# ============================================================
def plot_ke_settling():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    found = False
    for i, subdir in enumerate(["experiments/n200_serial_pairs",
                                 "experiments/n1000_serial_pairs"]):
        df = load(ROOT / subdir / "diagnostics.csv")
        if df is None: continue
        found = True
        N = 200 if i == 0 else 1000
        ax = axes[i]
        ax.plot(df['time'], df['kinetic_energy'], lw=1.8, color='steelblue', label='KE')
        ax.plot(df['time'], df['potential_energy'], lw=1.8, color='darkorange',
                ls='--', label='PE')
        ax.plot(df['time'], df['kinetic_energy'] + df['potential_energy'],
                lw=1.5, color='green', ls=':', label='Total E')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Energy (J)')
        ax.set_title(f'Energy vs Time (N={N})'); ax.legend()
    if not found: print("  [skip] experiment diagnostics"); return
    save(fig, "fig_ke_settling")

# ============================================================
#  9. Damping study
# ============================================================
def plot_damping():
    df = load(ROOT / "science_bonus/bounce_damping_scan.csv")
    if df is None: print("  [skip] bounce_damping_scan.csv"); return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['gamma_n'], df['final_KE'], 'o-', color='firebrick', ms=7)
    ax.set_xlabel(r'Damping coefficient $\gamma_n$ (kg/s)')
    ax.set_ylabel('Final Kinetic Energy (J)')
    ax.set_title('Effect of Damping on Residual KE')
    save(fig, "fig_damping")

# ============================================================
#  Main
# ============================================================
if __name__ == '__main__':
    print("Generating figures...")
    plot_freefall()
    plot_dt_error()
    plot_bounce()
    plot_profiling()
    plot_strong_scaling()
    plot_weak_scaling()
    plot_neighbor()
    plot_ke_settling()
    plot_damping()
    print(f"\nAll figures saved to {FIGS}/")
