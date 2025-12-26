import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from datetime import datetime
from scipy import stats

from imposter_llm_game import simulate_game, DEFAULT_MODEL


# Adjust this to change the base secret word and LLM model
SECRET_WORD = "piano"
MODEL = DEFAULT_MODEL

VERBOSE = True  # Set to False to reduce output
CONFIDENCE_LEVEL = 0.95  # For confidence intervals


def print_verbose(*args, **kwargs):
    """Print only if VERBOSE is True"""
    if VERBOSE:
        print(*args, **kwargs)


def calculate_convergence_time(result):
    """Calculate how many rounds it took to reach a winner"""
    if not result["votes"]:
        return result["params"]["max_rounds"]  # Reached max rounds
    return result["votes"][-1]["round"]


def calculate_statistics(df, metric_col, group_col):
    """Calculate mean, std, CI for a metric grouped by a column"""
    stats_df = df.groupby(group_col)[metric_col].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count'),
        ('sem', lambda x: stats.sem(x, nan_policy='omit'))
    ]).reset_index()
    
    # Calculate confidence intervals
    confidence = CONFIDENCE_LEVEL
    stats_df['ci'] = stats_df.apply(
        lambda row: stats.t.ppf((1 + confidence) / 2, row['count'] - 1) * row['sem']
        if row['count'] > 1 else 0,
        axis=1
    )
    
    return stats_df


def run_vary_N(
    Ns=(4, 6, 8, 10, 12),
    m: int = 1,
    rounds_before_vote: int = 1,
    max_rounds: int = 10,
    memory_length: int = 10,
    vote_memory_length: int = 10,
    num_reps: int = 15,  # Increased for better statistics
    base_seed: int = 1000,
):
    print_verbose(f"\n{'='*60}")
    print_verbose(f"EXPERIMENT 1: Varying N (number of players)")
    print_verbose(f"{'='*60}")
    print_verbose(f"Parameters: m={m}, rounds_before_vote={rounds_before_vote}, max_rounds={max_rounds}")
    print_verbose(f"Testing N values: {Ns}")
    print_verbose(f"Repetitions per N: {num_reps}")
    
    start_time = time.time()
    rows = []
    total_games = len(Ns) * num_reps
    game_count = 0
    
    for N in Ns:
        print_verbose(f"\n--- Testing N={N} ---")
        for rep in range(num_reps):
            game_count += 1
            seed = base_seed + 101 * N + rep
            print_verbose(f"  Game {game_count}/{total_games} (N={N}, rep={rep+1}/{num_reps})...", end=" ")
            
            game_start = time.time()
            res = simulate_game(
                secret_word=SECRET_WORD,
                N=N,
                m=m,
                rounds_before_vote=rounds_before_vote,
                max_rounds=max_rounds,
                memory_length=memory_length,
                vote_memory_length=vote_memory_length,
                model=MODEL,
                seed=seed,
                imposter_indices=None,  # random position
            )
            game_time = time.time() - game_start
            convergence_time = calculate_convergence_time(res)
            
            print_verbose(f"Winner: {res['winner']}, Rounds: {convergence_time} ({game_time:.1f}s)")
            rows.append(
                {
                    "N": N,
                    "rep": rep,
                    "winner": res["winner"],
                    "convergence_time": convergence_time,
                    "imposter_win": 1 if res["winner"] == "imposters" else 0,
                }
            )
    
    elapsed = time.time() - start_time
    print_verbose(f"\n✓ Completed in {elapsed:.1f}s ({elapsed/total_games:.1f}s per game)")
    
    df = pd.DataFrame(rows)
    df.to_csv("data_vary_N.csv", index=False)
    print_verbose("Saved raw data to data_vary_N.csv")
    
    # Calculate win rate statistics
    win_stats = calculate_statistics(df, 'imposter_win', 'N')
    win_stats.rename(columns={'mean': 'imposter_win_rate', 'ci': 'win_rate_ci'}, inplace=True)
    
    # Calculate convergence time statistics
    conv_stats = calculate_statistics(df, 'convergence_time', 'N')
    conv_stats.rename(columns={'mean': 'mean_convergence_time', 'ci': 'convergence_ci'}, inplace=True)
    
    # Convergence by winner type
    df_imposter_wins = df[df['winner'] == 'imposters']
    df_crew_wins = df[df['winner'] == 'crewmates']
    
    imposter_conv = calculate_statistics(df_imposter_wins, 'convergence_time', 'N') if len(df_imposter_wins) > 0 else None
    crew_conv = calculate_statistics(df_crew_wins, 'convergence_time', 'N') if len(df_crew_wins) > 0 else None
    
    if imposter_conv is not None:
        imposter_conv.rename(columns={'mean': 'imposter_conv_time', 'ci': 'imposter_conv_ci'}, inplace=True)
    if crew_conv is not None:
        crew_conv.rename(columns={'mean': 'crew_conv_time', 'ci': 'crew_conv_ci'}, inplace=True)
    
    summary = win_stats[['N', 'imposter_win_rate', 'win_rate_ci', 'count']].merge(
        conv_stats[['N', 'mean_convergence_time', 'convergence_ci']], on='N'
    )
    if imposter_conv is not None:
        summary = summary.merge(imposter_conv[['N', 'imposter_conv_time', 'imposter_conv_ci']], on='N', how='left')
    if crew_conv is not None:
        summary = summary.merge(crew_conv[['N', 'crew_conv_time', 'crew_conv_ci']], on='N', how='left')
    
    summary.to_csv("summary_vary_N.csv", index=False)
    print_verbose("Saved summary to summary_vary_N.csv")

    # Plot 1: Win rate with error bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.errorbar(summary["N"], summary["imposter_win_rate"], 
                 yerr=summary["win_rate_ci"], marker="o", capsize=5, capthick=2)
    ax1.set_xlabel("Number of players (N)")
    ax1.set_ylabel("Imposter win rate")
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Imposter win rate vs N (n={num_reps} games per condition)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax1.legend()
    
    # Plot 2: Convergence times
    ax2.errorbar(summary["N"], summary["mean_convergence_time"], 
                 yerr=summary["convergence_ci"], marker="o", capsize=5, capthick=2, label='Overall')
    if 'imposter_conv_time' in summary.columns:
        ax2.errorbar(summary["N"], summary["imposter_conv_time"], 
                     yerr=summary["imposter_conv_ci"], marker="s", capsize=5, capthick=2, 
                     label='Imposter wins', alpha=0.7)
    if 'crew_conv_time' in summary.columns:
        ax2.errorbar(summary["N"], summary["crew_conv_time"], 
                     yerr=summary["crew_conv_ci"], marker="^", capsize=5, capthick=2, 
                     label='Crewmate wins', alpha=0.7)
    ax2.set_xlabel("Number of players (N)")
    ax2.set_ylabel("Rounds to convergence")
    ax2.set_title("Game convergence time vs N")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("vary_N_results.png", dpi=200)
    plt.close()

    return df, summary


def run_vary_rounds_before_vote(
    rounds_list=(1, 2, 3),
    N: int = 8,
    m: int = 1,
    max_rounds: int = 12,
    memory_length: int = 10,
    vote_memory_length: int = 10,
    num_reps: int = 15,
    base_seed: int = 2000,
):
    print_verbose(f"\n{'='*60}")
    print_verbose(f"EXPERIMENT 2: Varying rounds_before_vote")
    print_verbose(f"{'='*60}")
    print_verbose(f"Parameters: N={N}, m={m}, max_rounds={max_rounds}")
    print_verbose(f"Testing rounds_before_vote values: {rounds_list}")
    print_verbose(f"Repetitions per value: {num_reps}")
    
    start_time = time.time()
    rows = []
    total_games = len(rounds_list) * num_reps
    game_count = 0
    
    for r in rounds_list:
        print_verbose(f"\n--- Testing rounds_before_vote={r} ---")
        for rep in range(num_reps):
            game_count += 1
            seed = base_seed + 113 * r + rep
            print_verbose(f"  Game {game_count}/{total_games} (rounds={r}, rep={rep+1}/{num_reps})...", end=" ")
            
            game_start = time.time()
            res = simulate_game(
                secret_word=SECRET_WORD,
                N=N,
                m=m,
                rounds_before_vote=r,
                max_rounds=max_rounds,
                memory_length=memory_length,
                vote_memory_length=vote_memory_length,
                model=MODEL,
                seed=seed,
            )
            game_time = time.time() - game_start
            convergence_time = calculate_convergence_time(res)
            
            print_verbose(f"Winner: {res['winner']}, Rounds: {convergence_time} ({game_time:.1f}s)")
            rows.append(
                {
                    "rounds_before_vote": r,
                    "rep": rep,
                    "winner": res["winner"],
                    "convergence_time": convergence_time,
                    "imposter_win": 1 if res["winner"] == "imposters" else 0,
                }
            )

    elapsed = time.time() - start_time
    print_verbose(f"\n✓ Completed in {elapsed:.1f}s ({elapsed/total_games:.1f}s per game)")

    df = pd.DataFrame(rows)
    df.to_csv("data_vary_rounds.csv", index=False)
    
    # Calculate statistics
    win_stats = calculate_statistics(df, 'imposter_win', 'rounds_before_vote')
    win_stats.rename(columns={'mean': 'imposter_win_rate', 'ci': 'win_rate_ci'}, inplace=True)
    
    conv_stats = calculate_statistics(df, 'convergence_time', 'rounds_before_vote')
    conv_stats.rename(columns={'mean': 'mean_convergence_time', 'ci': 'convergence_ci'}, inplace=True)
    
    df_imposter_wins = df[df['winner'] == 'imposters']
    df_crew_wins = df[df['winner'] == 'crewmates']
    
    imposter_conv = calculate_statistics(df_imposter_wins, 'convergence_time', 'rounds_before_vote') if len(df_imposter_wins) > 0 else None
    crew_conv = calculate_statistics(df_crew_wins, 'convergence_time', 'rounds_before_vote') if len(df_crew_wins) > 0 else None
    
    if imposter_conv is not None:
        imposter_conv.rename(columns={'mean': 'imposter_conv_time', 'ci': 'imposter_conv_ci'}, inplace=True)
    if crew_conv is not None:
        crew_conv.rename(columns={'mean': 'crew_conv_time', 'ci': 'crew_conv_ci'}, inplace=True)
    
    summary = win_stats[['rounds_before_vote', 'imposter_win_rate', 'win_rate_ci', 'count']].merge(
        conv_stats[['rounds_before_vote', 'mean_convergence_time', 'convergence_ci']], on='rounds_before_vote'
    )
    if imposter_conv is not None:
        summary = summary.merge(imposter_conv[['rounds_before_vote', 'imposter_conv_time', 'imposter_conv_ci']], 
                               on='rounds_before_vote', how='left')
    if crew_conv is not None:
        summary = summary.merge(crew_conv[['rounds_before_vote', 'crew_conv_time', 'crew_conv_ci']], 
                               on='rounds_before_vote', how='left')
    
    summary.to_csv("summary_vary_rounds.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.errorbar(summary["rounds_before_vote"], summary["imposter_win_rate"], 
                 yerr=summary["win_rate_ci"], marker="o", capsize=5, capthick=2)
    ax1.set_xlabel("Rounds before each vote")
    ax1.set_ylabel("Imposter win rate")
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Imposter win rate vs rounds_before_vote (n={num_reps})")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax1.legend()
    
    ax2.errorbar(summary["rounds_before_vote"], summary["mean_convergence_time"], 
                 yerr=summary["convergence_ci"], marker="o", capsize=5, capthick=2, label='Overall')
    if 'imposter_conv_time' in summary.columns:
        ax2.errorbar(summary["rounds_before_vote"], summary["imposter_conv_time"], 
                     yerr=summary["imposter_conv_ci"], marker="s", capsize=5, capthick=2, 
                     label='Imposter wins', alpha=0.7)
    if 'crew_conv_time' in summary.columns:
        ax2.errorbar(summary["rounds_before_vote"], summary["crew_conv_time"], 
                     yerr=summary["crew_conv_ci"], marker="^", capsize=5, capthick=2, 
                     label='Crewmate wins', alpha=0.7)
    ax2.set_xlabel("Rounds before each vote")
    ax2.set_ylabel("Rounds to convergence")
    ax2.set_title("Game convergence time vs rounds_before_vote")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("vary_rounds_results.png", dpi=200)
    plt.close()

    return df, summary


def run_vary_memory_length(
    memory_lengths=(None, 5, 10, 20),
    N: int = 8,
    m: int = 1,
    rounds_before_vote: int = 1,
    max_rounds: int = 10,
    vote_memory_length: int = 10,
    num_reps: int = 15,
    base_seed: int = 3000,
):
    print_verbose(f"\n{'='*60}")
    print_verbose(f"EXPERIMENT 3: Varying memory_length")
    print_verbose(f"{'='*60}")
    print_verbose(f"Parameters: N={N}, m={m}, rounds_before_vote={rounds_before_vote}")
    print_verbose(f"Testing memory_length values: {memory_lengths}")
    print_verbose(f"Repetitions per value: {num_reps}")
    
    start_time = time.time()
    rows = []
    total_games = len(memory_lengths) * num_reps
    game_count = 0
    
    for mem in memory_lengths:
        mem_str = "full" if mem is None else str(mem)
        print_verbose(f"\n--- Testing memory_length={mem_str} ---")
        for rep in range(num_reps):
            game_count += 1
            seed = base_seed + 127 * (0 if mem is None else mem) + rep
            print_verbose(f"  Game {game_count}/{total_games} (memory={mem_str}, rep={rep+1}/{num_reps})...", end=" ")
            
            game_start = time.time()
            res = simulate_game(
                secret_word=SECRET_WORD,
                N=N,
                m=m,
                rounds_before_vote=rounds_before_vote,
                max_rounds=max_rounds,
                memory_length=mem,
                vote_memory_length=vote_memory_length,
                model=MODEL,
                seed=seed,
            )
            game_time = time.time() - game_start
            convergence_time = calculate_convergence_time(res)
            
            print_verbose(f"Winner: {res['winner']}, Rounds: {convergence_time} ({game_time:.1f}s)")
            rows.append(
                {
                    "memory_length": -1 if mem is None else mem,
                    "rep": rep,
                    "winner": res["winner"],
                    "convergence_time": convergence_time,
                    "imposter_win": 1 if res["winner"] == "imposters" else 0,
                }
            )

    elapsed = time.time() - start_time
    print_verbose(f"\n✓ Completed in {elapsed:.1f}s ({elapsed/total_games:.1f}s per game)")

    df = pd.DataFrame(rows)
    df.to_csv("data_vary_memory.csv", index=False)
    
    win_stats = calculate_statistics(df, 'imposter_win', 'memory_length')
    win_stats.rename(columns={'mean': 'imposter_win_rate', 'ci': 'win_rate_ci'}, inplace=True)
    
    conv_stats = calculate_statistics(df, 'convergence_time', 'memory_length')
    conv_stats.rename(columns={'mean': 'mean_convergence_time', 'ci': 'convergence_ci'}, inplace=True)
    
    df_imposter_wins = df[df['winner'] == 'imposters']
    df_crew_wins = df[df['winner'] == 'crewmates']
    
    imposter_conv = calculate_statistics(df_imposter_wins, 'convergence_time', 'memory_length') if len(df_imposter_wins) > 0 else None
    crew_conv = calculate_statistics(df_crew_wins, 'convergence_time', 'memory_length') if len(df_crew_wins) > 0 else None
    
    if imposter_conv is not None:
        imposter_conv.rename(columns={'mean': 'imposter_conv_time', 'ci': 'imposter_conv_ci'}, inplace=True)
    if crew_conv is not None:
        crew_conv.rename(columns={'mean': 'crew_conv_time', 'ci': 'crew_conv_ci'}, inplace=True)
    
    summary = win_stats[['memory_length', 'imposter_win_rate', 'win_rate_ci', 'count']].merge(
        conv_stats[['memory_length', 'mean_convergence_time', 'convergence_ci']], on='memory_length'
    )
    if imposter_conv is not None:
        summary = summary.merge(imposter_conv[['memory_length', 'imposter_conv_time', 'imposter_conv_ci']], 
                               on='memory_length', how='left')
    if crew_conv is not None:
        summary = summary.merge(crew_conv[['memory_length', 'crew_conv_time', 'crew_conv_ci']], 
                               on='memory_length', how='left')
    
    summary.to_csv("summary_vary_memory.csv", index=False)

    # Create x-axis labels and values
    x_labels = []
    x_vals = []
    for x in summary["memory_length"]:
        if x == -1:
            x_labels.append("full")
            x_vals.append(0)
        else:
            x_labels.append(str(int(x)))
            x_vals.append(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.errorbar(x_vals, summary["imposter_win_rate"], 
                 yerr=summary["win_rate_ci"], marker="o", capsize=5, capthick=2)
    ax1.set_xlabel("Memory length (0 = full history)")
    ax1.set_ylabel("Imposter win rate")
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Imposter win rate vs memory_length (n={num_reps})")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax1.legend()
    
    ax2.errorbar(x_vals, summary["mean_convergence_time"], 
                 yerr=summary["convergence_ci"], marker="o", capsize=5, capthick=2, label='Overall')
    if 'imposter_conv_time' in summary.columns:
        ax2.errorbar(x_vals, summary["imposter_conv_time"], 
                     yerr=summary["imposter_conv_ci"], marker="s", capsize=5, capthick=2, 
                     label='Imposter wins', alpha=0.7)
    if 'crew_conv_time' in summary.columns:
        ax2.errorbar(x_vals, summary["crew_conv_time"], 
                     yerr=summary["crew_conv_ci"], marker="^", capsize=5, capthick=2, 
                     label='Crewmate wins', alpha=0.7)
    ax2.set_xlabel("Memory length (0 = full history)")
    ax2.set_ylabel("Rounds to convergence")
    ax2.set_title("Game convergence time vs memory_length")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("vary_memory_results.png", dpi=200)
    plt.close()

    return df, summary


def run_vary_imposter_position(
    N: int = 8,
    m: int = 1,
    rounds_before_vote: int = 1,
    max_rounds: int = 10,
    memory_length: int = 10,
    vote_memory_length: int = 10,
    num_reps: int = 15,
    base_seed: int = 4000,
):
    assert m == 1, "This sweep assumes a single imposter."

    print_verbose(f"\n{'='*60}")
    print_verbose(f"EXPERIMENT 4: Varying imposter position")
    print_verbose(f"{'='*60}")
    print_verbose(f"Parameters: N={N}, m={m}, rounds_before_vote={rounds_before_vote}")
    print_verbose(f"Testing positions: 0 to {N-1}")
    print_verbose(f"Repetitions per position: {num_reps}")
    
    start_time = time.time()
    rows = []
    positions = list(range(N))
    total_games = len(positions) * num_reps
    game_count = 0
    
    for pos in positions:
        print_verbose(f"\n--- Testing imposter at position {pos} ---")
        for rep in range(num_reps):
            game_count += 1
            seed = base_seed + 139 * pos + rep
            print_verbose(f"  Game {game_count}/{total_games} (pos={pos}, rep={rep+1}/{num_reps})...", end=" ")
            
            game_start = time.time()
            res = simulate_game(
                secret_word=SECRET_WORD,
                N=N,
                m=m,
                rounds_before_vote=rounds_before_vote,
                max_rounds=max_rounds,
                memory_length=memory_length,
                vote_memory_length=vote_memory_length,
                model=MODEL,
                seed=seed,
                imposter_indices=[pos],
            )
            game_time = time.time() - game_start
            convergence_time = calculate_convergence_time(res)
            
            print_verbose(f"Winner: {res['winner']}, Rounds: {convergence_time} ({game_time:.1f}s)")
            rows.append(
                {
                    "imposter_pos": pos,
                    "rep": rep,
                    "winner": res["winner"],
                    "convergence_time": convergence_time,
                    "imposter_win": 1 if res["winner"] == "imposters" else 0,
                }
            )

    elapsed = time.time() - start_time
    print_verbose(f"\n✓ Completed in {elapsed:.1f}s ({elapsed/total_games:.1f}s per game)")

    df = pd.DataFrame(rows)
    df.to_csv("data_vary_position.csv", index=False)
    
    win_stats = calculate_statistics(df, 'imposter_win', 'imposter_pos')
    win_stats.rename(columns={'mean': 'imposter_win_rate', 'ci': 'win_rate_ci'}, inplace=True)
    
    conv_stats = calculate_statistics(df, 'convergence_time', 'imposter_pos')
    conv_stats.rename(columns={'mean': 'mean_convergence_time', 'ci': 'convergence_ci'}, inplace=True)
    
    df_imposter_wins = df[df['winner'] == 'imposters']
    df_crew_wins = df[df['winner'] == 'crewmates']
    
    imposter_conv = calculate_statistics(df_imposter_wins, 'convergence_time', 'imposter_pos') if len(df_imposter_wins) > 0 else None
    crew_conv = calculate_statistics(df_crew_wins, 'convergence_time', 'imposter_pos') if len(df_crew_wins) > 0 else None
    
    if imposter_conv is not None:
        imposter_conv.rename(columns={'mean': 'imposter_conv_time', 'ci': 'imposter_conv_ci'}, inplace=True)
    if crew_conv is not None:
        crew_conv.rename(columns={'mean': 'crew_conv_time', 'ci': 'crew_conv_ci'}, inplace=True)
    
    summary = win_stats[['imposter_pos', 'imposter_win_rate', 'win_rate_ci', 'count']].merge(
        conv_stats[['imposter_pos', 'mean_convergence_time', 'convergence_ci']], on='imposter_pos'
    )
    if imposter_conv is not None:
        summary = summary.merge(imposter_conv[['imposter_pos', 'imposter_conv_time', 'imposter_conv_ci']], 
                               on='imposter_pos', how='left')
    if crew_conv is not None:
        summary = summary.merge(crew_conv[['imposter_pos', 'crew_conv_time', 'crew_conv_ci']], 
                               on='imposter_pos', how='left')
    
    summary.to_csv("summary_vary_position.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.errorbar(summary["imposter_pos"], summary["imposter_win_rate"], 
                 yerr=summary["win_rate_ci"], marker="o", capsize=5, capthick=2)
    ax1.set_xlabel("Imposter position in lineup (0 = first)")
    ax1.set_ylabel("Imposter win rate")
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Imposter win rate vs position (n={num_reps})")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax1.legend()
    
    ax2.errorbar(summary["imposter_pos"], summary["mean_convergence_time"], 
                 yerr=summary["convergence_ci"], marker="o", capsize=5, capthick=2, label='Overall')
    if 'imposter_conv_time' in summary.columns:
        ax2.errorbar(summary["imposter_pos"], summary["imposter_conv_time"], 
                     yerr=summary["imposter_conv_ci"], marker="s", capsize=5, capthick=2, 
                     label='Imposter wins', alpha=0.7)
    if 'crew_conv_time' in summary.columns:
        ax2.errorbar(summary["imposter_pos"], summary["crew_conv_time"], 
                     yerr=summary["crew_conv_ci"], marker="^", capsize=5, capthick=2, 
                     label='Crewmate wins', alpha=0.7)
    ax2.set_xlabel("Imposter position in lineup (0 = first)")
    ax2.set_ylabel("Rounds to convergence")
    ax2.set_title("Game convergence time vs position")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("vary_position_results.png", dpi=200)
    plt.close()

    return df, summary


def perform_statistical_tests(df, group_col, metric_col='imposter_win'):
    """Perform pairwise statistical tests between conditions"""
    print_verbose(f"\n{'='*60}")
    print_verbose(f"Statistical Significance Tests ({metric_col})")
    print_verbose(f"{'='*60}")
    
    groups = sorted(df[group_col].unique())
    results = []
    
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            data1 = df[df[group_col] == g1][metric_col].values
            data2 = df[df[group_col] == g2][metric_col].values
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(data1, data2)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
            else:
                cohens_d = 0
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            results.append({
                'comparison': f"{g1} vs {g2}",
                'mean_1': np.mean(data1),
                'mean_2': np.mean(data2),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significance': significance
            })
            
            print_verbose(f"{g1} vs {g2}: t={t_stat:.3f}, p={p_value:.4f} {significance}, d={cohens_d:.3f}")
    
    results_df = pd.DataFrame(results)
    print_verbose(f"\n*** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    
    return results_df


def main():
    print_verbose(f"\n{'#'*60}")
    print_verbose(f"# LLM Imposter Game Experiments")
    print_verbose(f"# Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_verbose(f"# Secret word: {SECRET_WORD}")
    print_verbose(f"# Model: {MODEL}")
    print_verbose(f"# Confidence level: {CONFIDENCE_LEVEL*100}%")
    print_verbose(f"{'#'*60}")
    
    overall_start = time.time()
    
    os.makedirs("results", exist_ok=True)
    os.chdir("results")
    print_verbose(f"\nSaving results to: {os.getcwd()}\n")

    # Experiment 1: Vary N
    df_N, summary_N = run_vary_N()
    print_verbose("\n" + "="*60)
    print_verbose("N sweep summary:")
    print_verbose(summary_N.to_string(index=False))
    print_verbose("="*60)
    
    # Statistical tests for N
    stat_tests_N = perform_statistical_tests(df_N, 'N', 'imposter_win')
    stat_tests_N.to_csv("statistical_tests_N.csv", index=False)

    # Experiment 2: Vary rounds before vote
    df_rounds, summary_rounds = run_vary_rounds_before_vote()
    print_verbose("\n" + "="*60)
    print_verbose("Rounds-before-vote sweep summary:")
    print_verbose(summary_rounds.to_string(index=False))
    print_verbose("="*60)
    
    stat_tests_rounds = perform_statistical_tests(df_rounds, 'rounds_before_vote', 'imposter_win')
    stat_tests_rounds.to_csv("statistical_tests_rounds.csv", index=False)

    # Experiment 3: Vary memory length
    df_mem, summary_mem = run_vary_memory_length()
    print_verbose("\n" + "="*60)
    print_verbose("Memory-length sweep summary:")
    print_verbose(summary_mem.to_string(index=False))
    print_verbose("="*60)
    
    stat_tests_mem = perform_statistical_tests(df_mem, 'memory_length', 'imposter_win')
    stat_tests_mem.to_csv("statistical_tests_memory.csv", index=False)

    # Experiment 4: Vary imposter position
    df_pos, summary_pos = run_vary_imposter_position()
    print_verbose("\n" + "="*60)
    print_verbose("Imposter-position sweep summary:")
    print_verbose(summary_pos.to_string(index=False))
    print_verbose("="*60)
    
    stat_tests_pos = perform_statistical_tests(df_pos, 'imposter_pos', 'imposter_win')
    stat_tests_pos.to_csv("statistical_tests_position.csv", index=False)
    
    overall_elapsed = time.time() - overall_start
    print_verbose(f"\n{'#'*60}")
    print_verbose(f"# All experiments completed!")
    print_verbose(f"# Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print_verbose(f"# Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_verbose(f"#")
    print_verbose(f"# Generated files:")
    print_verbose(f"#   - Raw data: data_vary_*.csv")
    print_verbose(f"#   - Summaries: summary_vary_*.csv")
    print_verbose(f"#   - Statistical tests: statistical_tests_*.csv")
    print_verbose(f"#   - Plots: vary_*_results.png")
    print_verbose(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
