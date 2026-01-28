"""
Protein Sequence Generation with Memory-aware Test-time Scaling
Version: 1.0.0
Author:
Date: 2025-11-13
Description: Enhanced protein sequence generation with dynamic linear temperature scheduling, dynamic linear edit_sequence,
             contrastive Learning (Relative preference is adopted to implement contrastive learning, and memory bank guidance.
"""

import numpy as np
import argparse
import torch
import os
import glob
import random
from evodiff.utils import Tokenizer
import pathlib
from sequence_models.datasets import UniRefDataset
from tqdm import tqdm
from evodiff.plot import aa_reconstruction_parity_plot
import pandas as pd
from evodiff.pretrained import CARP_38M, CARP_640M, D3PM_BLOSUM_38M, D3PM_BLOSUM_640M, D3PM_UNIFORM_38M, D3PM_UNIFORM_640M,\
                           OA_DM_640M, OA_DM_38M, LR_AR_38M, LR_AR_640M, ESM1b_650M
#  --- momst ---
import re
import pickle
import matplotlib.pyplot as plt
from reward import rank_normalize_scores
import seaborn as sns
from guidance_auditor import MultiObjectiveAuditor, plot_full_average_multi_objective_audit
#  --- momst ---

home = str(pathlib.Path.home())

def main():
    # set seeds
    _ = torch.manual_seed(0)
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='oa_dm_640M',
                        help='Choice of: carp_38M carp_640M esm1b_650M \
                                 oa_dm_38M oa_dm_640M lr_ar_38M lr_ar_640M d3pm_blosum_38M d3pm_blosum_640M d3pm_uniform_38M d3pm_uniform_640M')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    #parser.add_argument('out_fpath', type=str, nargs='?', default=os.getenv('PT_OUTPUT_DIR', '/tmp') + '/')
    parser.add_argument('--num-seqs', type=int, default=20)
    parser.add_argument('--penalty', type=float, default=None) # repetition penalty, commonly 1.2 is used
    parser.add_argument('--delete-prev',action='store_true')  # Will delete previous generated sequences
    parser.add_argument('--count', default=0, type=int) # Start new gen sequences from 0, this is when appending new seqs to files
    parser.add_argument('--scheme', default=None, type=str,
                        help='use train-sample valid-sample test-sample or random to generate samples not from model')
    parser.add_argument('--amlt', action='store_true')
    parser.add_argument('--random-baseline', action='store_true')
    args = parser.parse_args()

    data = UniRefDataset('data/uniref50/', 'train', structure=False, max_len=2048)
    data_valid = UniRefDataset('data/uniref50/', 'rtest', structure=False, max_len=2048)

    d3pm = False
    if args.model_type=='esm1b_650M':
        checkpoint = ESM1b_650M()
    elif args.model_type=='carp_38M':
        checkpoint = CARP_38M()
    elif args.model_type=='carp_640M':
        checkpoint = CARP_640M()
    elif args.model_type=='oa_dm_38M':
        checkpoint = OA_DM_38M()
    elif args.model_type=='oa_dm_640M':
        checkpoint = OA_DM_640M()
    elif args.model_type=='lr_ar_38M':
        checkpoint = LR_AR_38M()
    elif args.model_type=='lr_ar_640M':
        checkpoint = LR_AR_640M()
    elif args.model_type=='d3pm_blosum_38M':
        checkpoint = D3PM_BLOSUM_38M(return_all=True)
        d3pm=True
    elif args.model_type=='d3pm_blosum_640M':
        checkpoint = D3PM_BLOSUM_640M(return_all=True)
        d3pm=True
    elif args.model_type == 'd3pm_uniform_38M':
        checkpoint = D3PM_UNIFORM_38M(return_all=True)
        d3pm=True
    elif args.model_type == 'd3pm_uniform_640M':
        checkpoint = D3PM_UNIFORM_640M(return_all=True)
        d3pm=True
    else:
        raise Exception("Please select either carp_38M, carp_640M, esm1b_650M, oa_dm_38M, oa_dm_640M, lr_ar_38M, lr_ar_640M, d3pm_blosum_38M, d3pm_blosum_640M, d3pm_uniform_38M, or d3pm_uniform_640M. You selected:", args.model_type)

    if d3pm:
        model, collater, tokenizer, scheme, timestep, Q_bar, Q = checkpoint
    else:
        model, collater, tokenizer, scheme = checkpoint

    torch.cuda.set_device(args.gpus)
    device = torch.device('cuda:' + str(args.gpus))
    model = model.eval().to(device)

    # Out directories
    if args.amlt:
        home = os.getenv('AMLT_OUTPUT_DIR', '/tmp') + '/'
        top_dir = ''
        out_fpath = home
    else:
        home = str(pathlib.Path.home()) + '/Desktop/DMs/'
        top_dir = home
        if not args.random_baseline:
            out_fpath = home + args.model_type + '/'
        else:
            scheme='random'
            out_fpath = home + 'random-baseline/'

    if not os.path.exists(out_fpath):
        os.makedirs(out_fpath)

    data_top_dir = top_dir + 'data/'

    # Delete prev runs
    if args.delete_prev:
        filelist = glob.glob(out_fpath+'generated*')
        for file in filelist:
            os.remove(file)
            print("Deleting", file, "in", out_fpath)

    # Run generation
    if scheme == 'causal-mask':
        sample, string = generate_autoreg(model, tokenizer, samples=args.num_seqs, penalty=args.penalty, device=device)

    elif scheme == 'test-sample':
        string = generate_valid_subset(data_valid, samples=args.num_seqs)

    elif scheme == 'random':
        train_prob_dist = aa_reconstruction_parity_plot(home, out_fpath, 'placeholder.csv', gen_file=False)
        string = []
        for _ in tqdm(range(args.num_seqs)):
            r_idx = np.random.choice(len(data))
            seq_len = len(data[r_idx][0])  # randomly sample a sequence length from train data
            i_string = generate_random_seq(seq_len, train_prob_dist)
            print(i_string)
            string.append([i_string])
    else:
        string = []
        sample = []
        for _ in tqdm(range(args.num_seqs)):
            r_idx = np.random.choice(len(data))
            seq_len = len(data[r_idx][0])  # randomly sample a sequence length from train data

            if scheme == 'mask':
                i_sample, i_string = generate_oaardm(model, tokenizer, seq_len, penalty=args.penalty, batch_size=1, device=device)
            elif scheme == 'd3pm':
                i_sample, i_string = generate_d3pm(model, tokenizer, Q, Q_bar, timestep, seq_len, batch_size=1,
                                                   device=device)
            string.append(i_string)
            sample.append(i_sample)
    print("String", string)
    # Write list of sequences (string) to fasta and CSV
    with open(out_fpath + 'generated_samples_string.csv', 'w') as f:
        for _s in string:
            f.write(_s[0] + "\n")
    with open(out_fpath + 'generated_samples_string.fasta', 'w') as f:
        for i, _s in enumerate(string):
            f.write(">SEQUENCE_" + str(args.count+i) + "\n" + str(_s[0]) + "\n")

    # Plot distribution of generated samples
    aa_reconstruction_parity_plot(home, out_fpath, 'generated_samples_string.csv')

def parse_trajectory_file(file_path):

    data_list = []
    reward_pattern = re.compile(r"(\w+)=([-+]?\d*\.\d+|\d+)")

    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return []

    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip(): continue

            it_match = re.search(r"Iteration:(\d+)", line)
            if not it_match: continue
            it_num = int(it_match.group(1))

            rewards_found = reward_pattern.findall(line)
            reward_values = [float(v) for k, v in rewards_found]

            data_list.append({
                'location_info': (it_num, 0, 0),
                'rewards': reward_values
            })

    return data_list

def analyze_protein_multi_rewards(memory_bank, save_folder, reward_names=None, rewards_weights=None):
    data = []
    for item in memory_bank:
        it_num, edit_pos_idx, cand_num = item['location_info']
        reward_sum = sum(r * w for r, w in zip(item['rewards'], rewards_weights))
        robust_score = float(np.min(item['rewards']))
        data.append({
            'iteration': it_num,
            'edit_pos': edit_pos_idx,
            'rewards': item['rewards'],
            'robust_score': robust_score,
            'reward_sum': reward_sum,
        })
    df = pd.DataFrame(data)

    num_rewards = len(df['rewards'].iloc[0])
    reward_cols = [reward_names[i] if reward_names else f'R{i}' for i in range(num_rewards)]
    df[reward_cols] = pd.DataFrame(df['rewards'].tolist(), index=df.index)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # --- Multiple Rewards Comparison ---
    plt.figure(figsize=(12, 7))

    reward_stats = df.groupby('iteration')[reward_cols].agg(['mean', 'std']).reset_index()

    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']

    for i, col in enumerate(reward_cols):
        color = colors[i % len(colors)]
        ls = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]

        means = reward_stats[col]['mean']
        stds = reward_stats[col]['std']
        iters = reward_stats['iteration']

        plt.plot(iters, means, label=f'Mean {col}', color=color,
                 linestyle=ls, marker=marker, linewidth=2, markersize=6)

        plt.fill_between(iters,
                         means - stds,
                         means + stds,
                         color=color, alpha=0.15)


        for x, y in zip(iters, means):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=7, color=color)

    plt.title("Comparison of Multiple Reward Metrics over Iterations", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Reward Value (Raw Scores)", fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    multi_reward_plot_path = os.path.join(save_folder, "multi_rewards_analysis.png")
    plt.savefig(multi_reward_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_protein_memory(memory_bank, save_folder, seq_len, reward_names=None, rewards_weights=None):

    data = []
    for item in memory_bank:
        it_num, edit_pos_idx, cand_num = item['location_info']
        reward_sum = sum(r * w for r, w in zip(item['rewards'], rewards_weights))
        robust_score = float(np.min(item['rewards']))
        data.append({
            'iteration': it_num,
            'edit_pos': edit_pos_idx,
            'rewards': item['rewards'],
            'robust_score': robust_score,
            'reward_sum': reward_sum,
        })
    df = pd.DataFrame(data)

    num_rewards = len(df['rewards'].iloc[0])
    reward_cols = [reward_names[i] if reward_names else f'R{i}' for i in range(num_rewards)]
    df[reward_cols] = pd.DataFrame(df['rewards'].tolist(), index=df.index)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    plt.figure(figsize=(12, 6))

    target_metric = 'reward_sum'
    it_stats = df.groupby('iteration')[target_metric].agg(['mean', 'std', 'max']).reset_index()

    plt.plot(it_stats['iteration'], it_stats['max'], marker='s', ls='--', color='gold', label='Global Best (Survivor)')
    plt.plot(it_stats['iteration'], it_stats['mean'], marker='o', ls='-', color='dodgerblue',
             label='Batch Average (New Member)')
    plt.fill_between(it_stats['iteration'],
                     it_stats['mean'] - it_stats['std'],
                     it_stats['mean'] + it_stats['std'],
                     color='dodgerblue', alpha=0.2, label='±1 Std Dev')

    for i, r in it_stats.iterrows():
        plt.annotate(f"{r['max']:.2f}", (r['iteration'], r['max']), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8)

    plt.title(f"Evolution of {target_metric}: Survivors vs New Members", fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(alpha=0.3)

    evolution_plot_path = os.path.join(save_folder, f"iteration_{target_metric}.png")
    plt.savefig(evolution_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))

    heatmap_data = df.pivot_table(index='iteration',
                                  columns='edit_pos',
                                  values= 'reward_sum',
                                  aggfunc='mean')

    for p in range(seq_len):
        if p not in heatmap_data.columns:
            heatmap_data[p] = np.nan
    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)

    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False, cbar_kws={'label': 'Rewards Sum Score'})  # 'label': 'Avg Robust Score'
    plt.title("Sequence Edit Sensitivity Heatmap", fontsize=14)
    plt.xlabel("Amino Acid Position")
    plt.ylabel("Iteration")

    heatmap_plot_path = os.path.join(save_folder, "edit_sensitivity_heatmap.png")
    plt.savefig(heatmap_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def pareto_filter(rewards):
    """
    Identifies the Pareto Front (non-dominated solutions) from a set of rewards.
    Assumes higher rewards are better (rewards must be normalized or raw "higher is better" scores).

    Args:
        rewards (np.ndarray): Array of shape (N, M) where N is the number of solutions
                              and M is the number of objectives/metrics.

    Returns:
        np.ndarray: A boolean mask where True indicates the solution is on the Pareto Front.
    """
    is_dominated = np.zeros(rewards.shape[0], dtype=bool)
    N = rewards.shape[0]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            # Check if solution j dominates solution i:
            # j dominates i if: (j >= i on all metrics) AND (j > i on at least one metric)

            # Check 1: Is j better or equal to i on ALL metrics?
            is_better_or_equal = np.all(rewards[j, :] >= rewards[i, :])

            # Check 2: Is j STRICTLY better than i on at least one metric?
            is_strictly_better = np.any(rewards[j, :] > rewards[i, :])

            if is_better_or_equal and is_strictly_better:
                # If j dominates i, then i is NOT on the Pareto Front.
                is_dominated[i] = True
                break # Move to the next solution i since i is already marked as dominated

    # The Pareto Front consists of solutions that are not dominated
    return ~is_dominated


def update_memory_bank(memory_bank, iteration_num, edit_length_num, candidate_num, sequence, rewards, rewards_weights, tokenizer, memory_size=20):
    """
    Updates the memory bank with the current sequence and its reward.
    Stores positional amino acid probability distributions (one-hot for single sequence).
    Maintains top_k high/low reward items based on reward.
    IMPORTANT: The 'tokenizer' must be the same object used by the pre-trained model to ensure index alignment.
    """
    # Deduplication: skip if sequence already in memory
    if any(item['sequence'] == sequence for item in memory_bank):
        return  # Do not add duplicate

    seq_len = len(sequence)
    pos_probs = [] # List to hold the positional probability distribution for the *current* sequence

    for i in range(seq_len): # Iterate through each position in the sequence
        aa_probs_at_pos = [0.0] * 20  # Initialize a probability vector of length 20 for position i
        token_id = tokenizer.a_to_i[sequence[i]]  # Convert the amino acid character at position i to its corresponding token ID using the provided tokenizer
        if 0 <= token_id < 20:  # Check if the token_id is within the valid range for standard amino acids (0-19)
            aa_probs_at_pos[token_id] = 1.0  # Set the probability of the specific amino acid at this position to 1.0 (one-hot encoding)
        pos_probs.append(aa_probs_at_pos)  # Append the one-hot vector for position i to the list for this sequence

    # sum_score = float(np.sum(np.array(rewards) * np.array(rewards_weights)))

    # Add the new sequence's information as a dictionary to the memory bank
    memory_bank.append({
        'location_info': [iteration_num, edit_length_num, candidate_num],
        'sequence': sequence,   # The raw sequence string
        'rewards': list(rewards),       # The computed reward for this sequence
        'pos_probs': pos_probs,  # The list of one-hot probability vectors for each position
        'sum_score': 0.0 # 'sum_score': sum_score
    })

    all_rewards_raw = np.array([item['rewards'] for item in memory_bank])
    num_metrics = all_rewards_raw.shape[1]  # 3

    ranked_matrix = np.zeros_like(all_rewards_raw, dtype=np.float32)
    for d in range(num_metrics):
        ranked_matrix[:, d] = rank_normalize_scores(all_rewards_raw[:, d])

    final_scores = np.dot(ranked_matrix, np.array(rewards_weights))

    for i in range(len(memory_bank)):
        memory_bank[i]['sum_score'] = float(final_scores[i])

    elite_size = int(memory_size * 0.5)
    buffer_size = memory_size - elite_size

    sorted_by_score = sorted(memory_bank, key=lambda x: x['sum_score'], reverse=True)
    elites = sorted_by_score[:elite_size]
    elites_seqs = set(item['sequence'] for item in elites)

    remaining = [item for item in memory_bank if item['sequence'] not in elites_seqs]

    recent_samples = remaining[-buffer_size:] if len(remaining) > buffer_size else remaining

    new_bank = elites + recent_samples

    memory_bank[:] = sorted(new_bank, key=lambda x: x['sum_score'])


def update_memory_bank(memory_bank, iteration_num, edit_length_num, candidate_num, sequence, rewards, rewards_weights, tokenizer, memory_size=20):  # for ablation
    """
    Updates the memory bank with the current sequence and its reward.
    Stores positional amino acid probability distributions (one-hot for single sequence).
    Maintains top_k high/low reward items based on reward.
    IMPORTANT: The 'tokenizer' must be the same object used by the pre-trained model to ensure index alignment.
    """
    # Deduplication: skip if sequence already in memory
    if any(item['sequence'] == sequence for item in memory_bank):
        return  # Do not add duplicate

    seq_len = len(sequence)
    pos_probs = [] # List to hold the positional probability distribution for the *current* sequence

    for i in range(seq_len): # Iterate through each position in the sequence
        aa_probs_at_pos = [0.0] * 20  # Initialize a probability vector of length 20 for position i
        token_id = tokenizer.a_to_i[sequence[i]]  # Convert the amino acid character at position i to its corresponding token ID using the provided tokenizer
        if 0 <= token_id < 20:  # Check if the token_id is within the valid range for standard amino acids (0-19)
            aa_probs_at_pos[token_id] = 1.0  # Set the probability of the specific amino acid at this position to 1.0 (one-hot encoding)
        pos_probs.append(aa_probs_at_pos)  # Append the one-hot vector for position i to the list for this sequence

    # # robust_score = worst of the three (Pareto principle)
    # robust_score = float(np.min(rewards))

    # Add the new sequence's information as a dictionary to the memory bank
    memory_bank.append({
        'location_info': [iteration_num, edit_length_num, candidate_num],
        'sequence': sequence,   # The raw sequence string
        'rewards': list(rewards),       # The computed reward for this sequence
        'pos_probs': pos_probs,  # The list of one-hot probability vectors for each position
        'robust_score': 0.0 # 'sum_score': sum_score,
    })

    all_rewards_raw = np.array([item['rewards'] for item in memory_bank])
    num_metrics = all_rewards_raw.shape[1]  # 3

    ranked_matrix = np.zeros_like(all_rewards_raw, dtype=np.float32)
    for d in range(num_metrics):
        ranked_matrix[:, d] = rank_normalize_scores(all_rewards_raw[:, d])
    final_scores = np.dot(ranked_matrix, np.array(rewards_weights))
    for i in range(len(memory_bank)):
        memory_bank[i]['sum_score'] = float(final_scores[i])

    # --- Memory Consolidation: Keep only top_k items ---
    sorted_items = sorted(memory_bank, key=lambda x: x['sum_score'])

    # Calculate how many high-reward and low-reward items to keep (up to half of top_k each)
    k_low = min(memory_size // 2, len(sorted_items))
    k_high = min(memory_size // 2, len(sorted_items)-k_low)

    top_low = sorted_items[:k_low]  # Select the top keep_low low-reward items
    top_high = sorted_items[-k_high:] if k_high > 0 else []  # Select the top keep_high high-reward items

    # Update the memory bank in-place with the selected high and low reward items
    memory_bank[:] = top_low + top_high


def get_memory_guidance_for_position(memory_bank, position, reward_dim, percentile_high, percentile_low, device):  # momst
    """
    Calculates weighted guidance probabilities from memory bank for a specific sequence position.
    Uses sequence reward as weight for averaging.
    Returns positive guidance (high reward) and negative guidance (low reward) probabilities for that position.
    Checks for validity before returning guidance.
    """

    if not memory_bank:
        uniform = torch.ones(20, device=device) / 20.0
        return uniform, uniform

    reward_vals = np.array([item['rewards'][reward_dim] for item in memory_bank])

    norm_rewards = rank_normalize_scores(reward_vals)

    thresh_high = np.percentile(norm_rewards, percentile_high)
    thresh_low = np.percentile(norm_rewards, percentile_low)

    pos_guidance = torch.zeros(20, device=device, dtype=torch.float32)
    neg_guidance = torch.zeros(20, device=device, dtype=torch.float32)
    pos_total_weight  = 0.0
    neg_total_weight  = 0.0

    for i, item in enumerate(memory_bank):  # for item in memory_bank:
        if position >= len(item['pos_probs']):
            continue
        r_val = norm_rewards[i]
        aa_prob = torch.tensor(item['pos_probs'][position], device=device, dtype=torch.float32)

        if r_val >= thresh_high:
            weight = (r_val - thresh_high) + 1.0
            pos_guidance += weight * aa_prob
            pos_total_weight += weight
        elif r_val <= thresh_low:
            weight = (thresh_low - r_val) + 1.0
            neg_guidance += weight * aa_prob
            neg_total_weight += weight

    # Normalize using total weights
    eps = 1e-8
    if pos_total_weight > eps:
        pos_guidance = pos_guidance / (pos_guidance.sum() + eps)  # ensure sum=1
    else:
        pos_guidance = torch.ones(20, device=device, dtype=torch.float32) / 20.0

    if neg_total_weight > eps:
        neg_guidance = neg_guidance / (neg_guidance.sum() + eps)
    else:
        neg_guidance = torch.ones(20, device=device, dtype=torch.float32) / 20.0

    return pos_guidance, neg_guidance


def generate_oaardm_order_opt(model, tokenizer, seq_len, penalty=None, batch_size=20, device='cuda'):
    # Generate a random start string and convert to tokens
    all_aas = tokenizer.all_aas
    mask = tokenizer.mask_id
    # Start from mask
    sample = torch.zeros((batch_size, seq_len))+mask
    sample = sample.to(torch.long)
    sample = sample.to(device)
    loc = np.arange(seq_len)
    timestep = torch.tensor([0] * batch_size)  # placeholder but not used in model
    timestep = timestep.to(device)
    with torch.no_grad():
        for _ in loc:
            # Prob-based loc sampling
            prediction = model(sample, timestep) # output shape B x L x T
            p = torch.nn.functional.softmax(prediction[:, :, :len(all_aas) - 6], dim=-1) # normalize along L dim
            nonmask_loc = (sample != mask).unsqueeze(-1).expand(p.shape[0], p.shape[1], p.shape[2])
            p[nonmask_loc] = 0 # ignore tokens that have already been sampled
            idx = torch.argmax(p.view(1, -1), dim=-1)
            pos = torch.div(idx, p.shape[-1],rounding_mode='trunc')
            #aas = idx % p.shape[-1] # for argmax use this
            aas = torch.multinomial(p[0, pos], num_samples=1) # argmax looks bad, sample at each confident pos
            #print(idx, pos, aas)
            sample[:, pos] = aas
            #print("pos", pos.item(), [tokenizer.untokenize(s) for s in sample])
    untokenized = [tokenizer.untokenize(s) for s in sample]
    return sample, untokenized



def generate_oaardm(model, tokenizer, seq_len, penalty=None, batch_size=3, device='cuda'):
    # Generate a random start string and convert to tokens
    all_aas = tokenizer.all_aas
    mask = tokenizer.mask_id

    # Start from mask
    sample = torch.zeros((batch_size, seq_len))+mask
    sample = sample.to(torch.long)
    sample = sample.to(device)

    # Unmask 1 loc at a time randomly
    loc = np.arange(seq_len)
    np.random.shuffle(loc)
    with torch.no_grad():
        for i in tqdm(loc):
            timestep = torch.tensor([0] * batch_size) # placeholder but not called in model
            timestep = timestep.to(device)
            prediction = model(sample, timestep) #, input_mask=input_mask.unsqueeze(-1)) #sample prediction given input
            p = prediction[:, i, :len(all_aas)-6] # sample at location i (random), dont let it predict non-standard AA
            p = torch.nn.functional.softmax(p, dim=1) # softmax over categorical probs
            p_sample = torch.multinomial(p, num_samples=1)
            # Repetition penalty
            if penalty is not None: # ignore if value is None
                for j in range(batch_size): # iterate over each obj in batch
                    case1 = (i == 0 and sample[j, i+1] == p_sample[j]) # beginning of seq
                    case2 = (i == seq_len-1 and sample[j, i-1] == p_sample[j]) # end of seq
                    case3 = ((i < seq_len-1 and i > 0) and ((sample[j, i-1] == p_sample[j]) or (sample[j, i+1] == p_sample[j]))) # middle of seq
                    if case1 or case2 or case3:
                        #print("identified repeat", p_sample, sample[i-1], sample[i+1])
                        p[j, int(p_sample[j])] /= penalty # reduce prob of that token by penalty value
                        p_sample[j] = torch.multinomial(p[j], num_samples=1) # resample
            sample[:, i] = p_sample.squeeze()
            #print([tokenizer.untokenize(s) for s in sample]) # check that sampling correctly
    #print("final seq", [tokenizer.untokenize(s) for s in sample])
    untokenized = [tokenizer.untokenize(s) for s in sample]
    return sample, untokenized


def generate_oaardm_reward(model, tokenizer, seq_len, reward_model, penalty=None, batch_size=3, rep = 5, device='cuda'):
    # Generate a random start string and convert to tokens
    all_aas = tokenizer.all_aas
    mask = tokenizer.mask_id

    # Start from mask
    sample = torch.zeros((batch_size, seq_len))+mask
    sample = sample.to(torch.long)
    sample = sample.to(device)

    # Unmask 1 loc at a time randomly
    loc = np.arange(seq_len)
    loc_set = [set(loc) for i in range(batch_size)] # Location set

    with torch.no_grad():
        for count, kkk in tqdm(enumerate(loc)):
            timestep = torch.tensor([0] * batch_size) # placeholder but not called in model
            timestep = timestep.to(device)
            prediction = model(sample, timestep) #, input_mask=input_mask.unsqueeze(-1)) #sample prediction given input
            
            ### Make Candidates 
            next_candidate = []
            pes_index_list = []
            for jjj in range(rep):
                loc_list = np.random.randint(len(loc_set[0]), size= batch_size) # Choose random location
                pes_index = [list(loc_set[i])[loc_list[i]]  for i in range(batch_size)] # Which position
                pes_index_list.append(pes_index)
                p = torch.stack([ prediction[i, pes_index[i], 0:20] for i in range(batch_size) ] )# sample at location i (random), dont let it predict non-standard AA
                p = torch.nn.functional.softmax(p, dim=1)
                p_sample = torch.multinomial(p, num_samples = 1)
                sample_fake = sample.clone()
                for iii in range(batch_size):
                    sample_fake[iii, pes_index[iii]] = p_sample.squeeze()[iii]
                next_candidate.append(sample_fake.clone())
            
            ### Calculate Reward
            if count % 1 ==0:
                reward_list = np.zeros((batch_size, rep))
                for jjj in range(rep):
                    prediction = model(next_candidate[jjj], timestep)
                    next_seq = next_candidate[jjj] * (next_candidate[jjj]!=28) + torch.argmax(prediction[:, :, 0:20], dim =2) * (next_candidate[jjj]==28)
                    reward_hoge = reward_model.stability_reward(next_seq)
                    reward_list[:,jjj] = reward_hoge
                next_index = np.argmax(reward_list, 1)
                print(np.max(reward_list,1))
            else:
                next_index = 0
                
            next_candidate = torch.stack(next_candidate)
            sample = torch.stack([ next_candidate[next_index[i],i,:] for i in range(batch_size) ] )
            for jjj in range(batch_size):
                loc_set[jjj].remove(pes_index_list[next_index[jjj]][jjj])
            #print([tokenizer.untokenize(s) for s in sample]) # check that sampling correctly
    #print("final seq", [tokenizer.untokenize(s) for s in sample])
    untokenized = [tokenizer.untokenize(s) for s in sample]
    return sample, untokenized


@torch.no_grad()
def generate_oaardm_reward_metrics(
        model,
        tokenizer,
        seq_len,
        reward_model,
        ori_pdb_file_path,
        batch,
        mask_for_loss,
        repeat_num,
        candidate,
        device,
):
    # Generate a random start string and convert to tokens
    mask = tokenizer.mask_id

    # Start from mask
    sample = torch.zeros((repeat_num, seq_len))+mask
    sample = sample.to(torch.long)
    sample = sample.to(device)  # bs * length

    # Unmask 1 loc at a time randomly
    loc = np.arange(seq_len)
    loc_set = [set(loc) for i in range(repeat_num)] # Location set

    with torch.no_grad():
        for count, kkk in enumerate(tqdm(loc)):
            timestep = torch.tensor([0] * repeat_num)  # placeholder but not called in model
            timestep = timestep.to(device)
            prediction = model(sample, timestep)  # bs * seq len * dim

            ### Make Candidates
            next_candidate = []
            pes_index_list = []
            for jjj in range(candidate):
                loc_list = np.random.randint(len(loc_set[0]), size=repeat_num)  # Choose random location
                pes_index = [list(loc_set[i])[loc_list[i]] for i in range(repeat_num)]  # Which position
                pes_index_list.append(pes_index)
                p = torch.stack([prediction[i, pes_index[i], 0:20] for i in range(
                    repeat_num)])  # sample at location i (random), dont let it predict non-standard AA
                p = torch.nn.functional.softmax(p, dim=1)  # bs * 20
                p_sample = torch.multinomial(p, num_samples=1)  # bs * 1
                sample_fake = sample.clone()
                for iii in range(repeat_num):
                    sample_fake[iii, pes_index[iii]] = p_sample.squeeze()[iii]
                next_candidate.append(sample_fake.clone())  # Shape Next candidate: Rep * Batch size * Seq_length

            ### reward & selection
            reward_list = np.zeros((repeat_num, candidate))
            for jjj in range(candidate):
                prediction = model(next_candidate[jjj], timestep)
                next_seq = next_candidate[jjj] * (next_candidate[jjj] != 28) + torch.argmax(prediction[:, :, 0:20],
                                                                                            dim=2) * (
                                       next_candidate[jjj] == 28)
                _, reward_hoge, _ = reward_model.reward_metrics(
                    protein_name=batch,
                    mask_for_loss=mask_for_loss,
                    S_sp=next_seq,
                    ori_pdb_file=ori_pdb_file_path,
                )
                reward_list[:, jjj] = reward_hoge
            next_index = np.argmax(reward_list, 1)
            
            # update sample
            next_candidate = torch.stack(next_candidate)
            sample = torch.stack([next_candidate[next_index[i], i, :] for i in range(repeat_num)])
            for jjj in range(repeat_num):
                loc_set[jjj].remove(pes_index_list[next_index[jjj]][jjj])

        untokenized = [tokenizer.untokenize(s) for s in sample]
        # pseudo_reward = reward_model.cal_rmsd_reward(sample)

    return sample, untokenized

@torch.no_grad()
def likelihood(
    model, tokenizer,seq_len, tokenized_sample, repeat_num, device
    ):
    tokenized_sample = tokenized_sample
    mask = tokenizer.mask_id
    likelihood = torch.tensor([0.0] * repeat_num).to(device)
    timestep = torch.tensor([0] * repeat_num)  # placeholder but not called in model
    timestep = timestep.to(device)
    for k in range(seq_len):
        generated_sequence_mask = tokenized_sample.clone()
        generated_sequence_mask[:,k] = mask 
        prediction = model(generated_sequence_mask, timestep)  # bs * seq len * dim
        p = torch.nn.functional.softmax(prediction, dim=1)
        likelihood += torch.log(p[ [i for i in range(repeat_num)], k, tokenized_sample[:,k]])
    return(likelihood.detach().cpu().numpy())


def generate_oaardm_reward_metrics_edit(
        model,
        tokenizer,
        seq_len,
        reward_model,
        ori_pdb_file_path,
        batch,
        mask_for_loss,
        repeat_num,
        candidate,
        folder_path,
        device,
        ss_option=False,
        iteration=15,
        edit_seqlength=5
):
    # Generate a random start string and convert to tokens
    mask = tokenizer.mask_id

    # Start from mask
    sample = torch.zeros((repeat_num, seq_len)) + mask
    sample = sample.to(torch.long)
    sample = sample.to(device)  # bs * length

    # Unmask 1 loc at a time randomly
    loc = np.arange(seq_len)
    loc_set = [set(loc) for i in range(repeat_num)]  # Location set

    with torch.no_grad():
        for ttt in tqdm(range(iteration)):

            if ttt == 0:
                length_edit = seq_len
            else:
                pass
            for count, kkk in enumerate(tqdm(range(length_edit))):
                timestep = torch.tensor([0] * repeat_num)  # placeholder but not called in model
                timestep = timestep.to(device)
                prediction = model(sample, timestep)  # bs * seq len * dim

                ### Make Candidates
                next_candidate = []
                pes_index_list = []
                for jjj in range(candidate):
                    loc_list = np.random.randint(len(loc_set[0]), size=repeat_num)  # Choose random location
                    pes_index = [list(loc_set[i])[loc_list[i]] for i in range(repeat_num)]  # Which position
                    pes_index_list.append(pes_index)
                    p = torch.stack([prediction[i, pes_index[i], 0:20] for i in range(
                        repeat_num)])  # sample at location i (random), dont let it predict non-standard AA
                    p = torch.nn.functional.softmax(p, dim=1)  # bs * 20
                    p_sample = torch.multinomial(p, num_samples=1)  # bs * 1
                    sample_fake = sample.clone()
                    for iii in range(repeat_num):
                        sample_fake[iii, pes_index[iii]] = p_sample.squeeze()[iii]
                    next_candidate.append(sample_fake.clone())  # Shape Next candidate: Rep * Batch size * Seq_length

                ### reward & selection
                reward_list = np.zeros((repeat_num, candidate))
                for jjj in range(candidate):
                    prediction = model(next_candidate[jjj], timestep)
                    next_seq = next_candidate[jjj] * (next_candidate[jjj] != 28) + torch.argmax(prediction[:, :, 0:20],
                                                                                                dim=2) * (
                                       next_candidate[jjj] == 28)

                    _, reward_hoge, _ = reward_model.reward_metrics(
                        protein_name=batch,
                        mask_for_loss=mask_for_loss,
                        S_sp=next_seq,
                        ori_pdb_file=ori_pdb_file_path,
                    )
                    reward_list[:, jjj] = reward_hoge
                next_index = np.argmax(reward_list, 1)

                # update sample
                next_candidate = torch.stack(next_candidate)
                sample = torch.stack([next_candidate[next_index[i], i, :] for i in range(repeat_num)])
                for jjj in range(repeat_num):
                    if loc_set[jjj] != []:
                        loc_set[jjj].remove(pes_index_list[next_index[jjj]][jjj])

            _, reward_hoge, position_list = reward_model.reward_metrics(
                protein_name=batch,
                mask_for_loss=mask_for_loss,
                S_sp=sample,
                ori_pdb_file=ori_pdb_file_path,
                save_pdb=True,
                add_info=ttt
            )

            print(reward_hoge)
            with open(folder_path + '/trajectory.txt', 'a') as f:
                f.write(', '.join(map(str, reward_hoge)) + '\n')

            if ttt == iteration - 1:
                untokenized = [tokenizer.untokenize(s) for s in sample]
                return sample, untokenized

            reward_hoge = np.exp(np.array(reward_hoge) * 5)  ##np.exp(reward_hoge*4) #Transform
            reward_hoge = np.nan_to_num(reward_hoge, nan=0.0)
            sampled_values = np.random.choice([i for i in range(repeat_num)], size=repeat_num,
                                              p=reward_hoge / np.sum(reward_hoge))

            sample = sample[sampled_values, :]
            '''
            if ss_option == True: 
                ## Resampling
                position_list = position_list[sampled_values, :]
                # Get location
                loc_set = [ [i for i,j in enumerate(position_list[kkk]) if j == True ] for kkk in range(repeat_num)] # Location set
                len_set = [len(kkk) for kkk in loc_set]
                length_edit = min(len_set)
                loc_set = [ random.sample(kkk, length_edit) for kkk in loc_set ]
                for iii in range(repeat_num):
                    sample[iii, loc_set[iii]] = mask
            else:
            '''
            loc_set = [random.sample(range(0, seq_len), edit_seqlength) for i in range(repeat_num)]  # Location set
            length_edit = edit_seqlength
            for iii in range(repeat_num):
                sample[iii, loc_set[iii]] = mask

@torch.no_grad()
def generate_oaardm_reward_metrics_edit_momost(
        model,
        tokenizer,
        seq_len,
        reward_model,
        ori_pdb_file_path,
        batch,
        mask_for_loss,
        repeat_num,
        candidate,
        folder_path,
        device,
        reward_metrics_name,
        rewards_weights,
        ss_option = False,
        iteration = 15,
        edit_seqlength = 5,
        memory_file_path = "protein_memorybank.pkl",
        memory_size_limit = 20,
):
    """
        Modified function incorporating Memory-aware Test-time Scaling concepts.
    """

    # Generate a random start string and convert to tokens
    mask = tokenizer.mask_id  # mask: {int64()} 28

    # --- Memory Initialization ---
    # Load existing memory or initialize empty
    try:
        with open(memory_file_path, 'rb') as f:
            memory_bank = pickle.load(f)
        print(f"Loaded memory bank with {len(memory_bank)} items.")
    except FileNotFoundError:
        print("Memory file not found, initializing empty memory bank.")
        memory_bank = []  # List of dictionaries: {'sequence_fragment': str, 'properties': dict, 'reward': float, 'context': str}

    # Start from mask
    sample = torch.zeros((repeat_num, seq_len))+mask  # [batch, seq_len] 全mask
    sample = sample.to(torch.long)
    sample = sample.to(device)  # bs * length

    # Unmask 1 loc at a time randomly
    loc = np.arange(seq_len)  # loc id
    loc_set = [set(loc) for i in range(repeat_num)] # Location set

    # momst for  temperature schedule
    temperature_schedule = lambda t: 10.0 * (1 - t / iteration) + 1.0 * (t / iteration)

    best_samples_global = None
    best_rewards_global = None
    best_rewards_array = None

    with torch.no_grad():
        for ttt in tqdm(range(iteration)):

            if ttt == 0:
                length_edit = seq_len
            else:
                pass

            current_temperature = temperature_schedule(ttt)

            for count, kkk in enumerate(tqdm(range(length_edit))):
                timestep = torch.tensor([0] * repeat_num)  # placeholder but not called in model
                timestep = timestep.to(device)
                prediction = model(sample, timestep)  # bs * seq len * dim

                ### Make Candidates
                next_candidate = []
                pes_index_list = []
                for jjj in range(candidate):
                    loc_list = np.random.randint(len(loc_set[0]), size=repeat_num)  # Choose random location
                    pes_index = [list(loc_set[i])[loc_list[i]] for i in range(repeat_num)]  # Which position
                    pes_index_list.append(pes_index)
                    p = torch.stack([prediction[i, pes_index[i], 0:20] for i in range(
                        repeat_num)])  # sample at location i (random), dont let it predict non-standard AA, [batch, 20] 20 aa type
                    # p = torch.nn.functional.softmax(p, dim=1)  # bs * 20
                    p = torch.nn.functional.softmax(p / current_temperature, dim=1)  # momst
                    _, p_top_idx = torch.max(p, dim=-1)

                    # --- Corrected Memory Guidance ---
                    p_combined = p.clone()  # [batch, 20]
                    batch_step_conflicts = []  # [batch, 20]
                    for batch_idx in range(repeat_num):
                        pos_idx = pes_index[batch_idx]
                        p_top_aa = p_top_idx[batch_idx]  # for analysis

                        # ③ Relative preference is adopted to implement contrastive learning, p_new = p_prior ⋅ exp(λ ⋅ log(p_pos/p_neg))), by momst
                        s_total = torch.zeros(20, device=device)
                        for r_dim in range(len(rewards_weights)):
                            p_pos, p_neg = get_memory_guidance_for_position(
                                memory_bank,
                                pos_idx,
                                reward_dim=r_dim,
                                percentile_high =50,
                                percentile_low=50,
                                device=device
                            )

                            # ③ Relative preference is adopted to implement contrastive learning, p_new = p_prior ⋅ exp(λ ⋅ log(p_pos/p_neg))), by momst
                            s_r = torch.log(p_pos.clamp(min=1e-8)) - torch.log(p_neg.clamp(min=1e-8))  # shape: (20,)
                            s_r = torch.clamp(s_r, min=-10, max=10)
                            s_total += rewards_weights[r_dim] * s_r
                        lam = 0.2  # ③
                        log_p_final = torch.log(p[batch_idx].clamp(min=1e-8)) + lam * s_total
                        # log_p_final = (torch.log(p[batch_idx].clamp(min=1e-8)) + lam * s_total) / current_temperature  # ← 这里是 element-wise multiplication!
                        log_p_shifted = log_p_final - torch.max(log_p_final, dim=-1, keepdim=True).values
                        p_combined[batch_idx] = torch.exp(log_p_shifted)

                        # Ensure probabilities sum to 1 (with safety check)
                        sum_p = torch.sum(p_combined[batch_idx])
                        if sum_p > 0:
                            p_combined[batch_idx] = p_combined[batch_idx] / sum_p
                        else:
                            # Fallback to uniform distribution if sum is zero or invalid
                            p_combined[batch_idx] = torch.ones_like(p_combined[batch_idx]) / len(p_combined[batch_idx])

                    p_sample = torch.multinomial(p_combined , num_samples=1)  # bs * 1
                    sample_fake = sample.clone()
                    for iii in range(repeat_num):
                        sample_fake[iii, pes_index[iii]] = p_sample.squeeze()[iii]
                    next_candidate.append(sample_fake.clone())  # Shape Next candidate: Rep * Batch size * Seq_length
                    del p, p_combined, p_sample, sample_fake
                del prediction

                ### reward & selection
                reward_multi_lists = np.zeros((repeat_num, candidate, len(rewards_weights)))
                # reward_list = np.zeros((repeat_num, candidate))
                reward_candidate_cache = []
                for jjj in range(candidate):
                    prediction = model(next_candidate[jjj], timestep)
                    next_seq = next_candidate[jjj] * (next_candidate[jjj] != 28) + torch.argmax(prediction[:, :, 0:20],
                                                                                                dim=2) * (
                                        next_candidate[jjj] == 28)


                    reward_multi, reward_hoge, _, _ = reward_model.reward_metrics(
                                protein_name=batch,
                                mask_for_loss=mask_for_loss,
                                S_sp= next_seq,
                                ori_pdb_file=ori_pdb_file_path,
                            )
                    reward_multi_lists[:, jjj, :] = reward_multi
                    # reward_list[:, jjj] = reward_hoge

                    reward_candidate_cache.append(reward_multi)  # [candidate, num_obj]
                    # # --- momst ---

                    # --- For memory update ---
                    next_seq_str = [tokenizer.untokenize(s) for s in next_seq]
                    reward_multi_list = list(reward_multi)
                    for iii in range(repeat_num):
                        update_memory_bank(memory_bank, ttt, pes_index_list[jjj][iii], jjj, next_seq_str[iii], reward_multi_list[iii], rewards_weights, tokenizer,
                                           memory_size=memory_size_limit)
                    del next_seq

                # --- Perform rank normalization at the batch level by momost ---
                reward_multi_lists_array = np.array(reward_multi_lists)
                ranked_rewards_candidate = np.zeros_like(reward_multi_lists, dtype=np.float32)
                for i in range(repeat_num):
                    for j in range(len(rewards_weights)):
                        ranked_rewards_candidate[i, :, j] = rank_normalize_scores(reward_multi_lists_array[i, :, j])
                ranked_rewards_candidate_agg = np.sum(ranked_rewards_candidate * np.array(rewards_weights), axis=-1)
                next_index = np.argmax(ranked_rewards_candidate_agg, 1)
                # # --- momost ---
                # next_index = np.argmax(reward_list, 1)

                # update sample
                next_candidate = torch.stack(next_candidate)
                sample = torch.stack([next_candidate[next_index[i], i, :] for i in range(repeat_num)])
                for jjj in range(repeat_num):
                    if loc_set[jjj]!=[]:
                        loc_set[jjj].remove(pes_index_list[next_index[jjj]][jjj])

            multi_rewards, reward_hoge, all_reward_dicts, _ = reward_model.reward_metrics(
                protein_name=batch,
                mask_for_loss=mask_for_loss,
                S_sp=sample,  # 966 line
                ori_pdb_file=ori_pdb_file_path,
                save_pdb=True,
                add_info=ttt
            )
            print(multi_rewards)  # for  observing
            rewards_array = np.array(multi_rewards)  # shape: (N, 3)

            # --- Perform rank normalization at the batch level by momost ---
            ranked_rewards = np.stack([rank_normalize_scores(rewards_array[:, i]) for i in range(len(rewards_weights))], axis=1).astype(np.float32)
            # ranked_agg = np.sum(ranked_rewards * np.array(rewards_weights), axis=1)
            # --- momost ---

            # Extract the normalized reward matrix and samples of Pareto frontier samples
            pareto_indices = np.where(pareto_filter(ranked_rewards))[0]
            pareto_ranked_rewards = ranked_rewards[pareto_indices, :]
            pareto_noranked_rewards = rewards_array[pareto_indices, :]
            pareto_sample = sample[pareto_indices, :]
            # pareto_ranked_agg = ranked_agg[pareto_indices]
            pareto_noranked_agg = np.sum(pareto_noranked_rewards * np.array(rewards_weights), axis=1)

            # # for single-objective no pareto filter
            # pareto_noranked_rewards = rewards_array
            # pareto_sample = sample
            # pareto_noranked_agg = np.sum(pareto_noranked_rewards * np.array(rewards_weights), axis=1)
            # --- momost ---

            # # --- Pareto-aware resampling using worst-case score by momost ---
            # robust_score = rewards_array.min(axis=1)  # shape: (N,)
            # robust_score = pareto_ranked_rewards.min(axis=1)  # shape: (N,)
            min_dim_indices = np.argmin(pareto_ranked_rewards, axis=-1)
            row_indices = np.arange(pareto_noranked_rewards.shape[0])  # 使用高级索引获取对应的原始奖励值
            robust_score = pareto_noranked_rewards[row_indices, min_dim_indices]

            # # for single-objective
            # robust_score = pareto_noranked_agg
            # --- momost ---

            # --- Elites retain core logic ---
            if ttt!=0:
                # The newly generated samples and their scores in the current round
                current_new_sample = pareto_sample.clone()
                current_rewards_array = pareto_noranked_rewards
                current_new_rewards = np.sum(current_rewards_array * np.array(rewards_weights), axis=-1)

                # Samples in the warehouse and their scores
                top_mem_samples = sorted(memory_bank, key=lambda x: x['sum_score'])[-min(10, len(memory_bank)):]
                mem_seqs = [top_mem_samples[i]['sequence'] for i in range(len(top_mem_samples))]
                memory_samples = torch.stack([torch.tensor([tokenizer.a_to_i[aa] for aa in s], device=device) for s in mem_seqs])
                mem_rewards_array = np.array([top_mem_samples[i]['rewards'] for i in range(len(top_mem_samples))])
                memory_rewards = np.sum(mem_rewards_array * np.array(rewards_weights), axis=-1)

                # --- Elites retain the core logic ---
                # Combine the elites from the previous round and the new samples from this round for global selection.
                all_samples = torch.cat([best_samples_global, current_new_sample, memory_samples], dim=0)
                all_rewards_array = np.concatenate([best_rewards_array, current_rewards_array, mem_rewards_array], axis=0)
                all_rewards = np.concatenate([best_rewards_global, current_new_rewards, memory_rewards], axis=0)

                # --- Ultimate simplification and deduplication in the elite retention block ---
                _, unique_indices = np.unique(all_samples.cpu().numpy(), axis=0, return_index=True)
                pool_samples = all_samples[unique_indices]
                pool_rewards_array = all_rewards_array[unique_indices]
                pool_rewards = all_rewards[unique_indices]

                # Sort by reward value and select the best repeat_num ones.
                sorted_indices = np.argsort(pool_rewards)[::-1]
                selected_indices = np.array(sorted_indices[:repeat_num]).copy()
                final_samples = pool_samples[selected_indices, :]
                final_rewards_array = pool_rewards_array[selected_indices, :]
                final_rewards = pool_rewards[selected_indices]
                pareto_sample = final_samples.clone()  # for Sampling
                pareto_noranked_rewards = final_rewards_array.copy()  # for trajectory.txt
                pareto_noranked_agg = final_rewards  # for Global Mean Sampling
                # Obtain robust_score
                final_ranked_rewards = np.stack([rank_normalize_scores(final_rewards_array[:, i]) for i in range(len(rewards_weights))], axis=1).astype(np.float32)
                min_dim_indices = np.argmin(final_ranked_rewards, axis=-1)
                row_indices = np.arange(final_rewards_array.shape[0])  # # Using advanced indexing to obtain the corresponding original reward values
                robust_score = final_rewards_array[row_indices, min_dim_indices]

            # --- recording best samples of current iteration ---
            if ttt == 0:
                best_samples_global = pareto_sample.clone()
                best_rewards_array = pareto_noranked_rewards  # best_rewards_array = rewards_array[sampled_indices, :]
                best_rewards_global = pareto_noranked_agg
            else:
                # Record the optimal score of this round as the benchmark for the next round.
                best_samples_global = pareto_sample.clone()  # pareto_sample.clone(), Either is fine. The elite part later has deduplication.
                best_rewards_array = final_rewards_array  # best_rewards_array = rewards_array[sampled_indices, :]
                best_rewards_global = final_rewards

            # --- momost ---
            current_sequences = [tokenizer.untokenize(s) for s in pareto_sample]
            with open(folder_path + '/trajectory.txt', 'a') as f:
                for seq_str, reward_values in zip(current_sequences, pareto_noranked_rewards):
                    reward_details = ", ".join([f"{name}={value:.4f}" for name, value in zip(reward_metrics_name.split(','), reward_values)])
                    f.write(f"Iteration:{ttt} | {reward_details} | Sequence:{seq_str}\n")
                f.write("\n")

            if ttt == iteration-1:
                # momst for saving memory bank
                memory_bank_path = os.path.join(folder_path, memory_file_path)
                with open(memory_bank_path, 'wb') as f:
                    pickle.dump(memory_bank, f)

                # Also save it as a text file for human observation
                memory_txt_path = memory_bank_path.replace('.pkl', '.txt')
                with open(memory_txt_path, 'w', encoding='utf-8') as f:
                    f.write("Location_Info\tReward\tSequence\tPosition_Probabilities\n")
                    for item in memory_bank:
                        location_info = item['location_info']
                        rewards = item['rewards']
                        sequence = item['sequence']
                        pos_probs_str = str(item['pos_probs']).replace(' ', '').replace('\n', '')
                        f.write(f"{location_info}\t{rewards}\t{sequence}\t{pos_probs_str}\n")
                print(f"Saved memory bank with {len(memory_bank)} items at end of function.")

                untokenized = [tokenizer.untokenize(s) for s in pareto_sample]  # untokenized = [tokenizer.untokenize(s) for s in sample]

                # # Warehouse protein analysis
                # medias_folder = os.path.join(folder_path, "medias")
                # os.makedirs(medias_folder, exist_ok=True)
                # analyze_protein_memory(memory_bank, medias_folder, seq_len=seq_len, reward_names=reward_metrics_name.split(','), rewards_weights=rewards_weights)
                # analyze_protein_multi_rewards(memory_bank, medias_folder, reward_names=reward_metrics_name.split(','),
                #                        rewards_weights=rewards_weights)

                return sample, untokenized


            # --- Resampling ---
            # # ① Global Mean Sampling: Resampling Based on the Mean of Multiple Rewards.
            # ranked_reward_hoge = np.exp(np.array(reward_hoge)*5)  # ranked_reward_hoge = np.exp(np.array(pareto_noranked_agg)*5)  # 指数变换放大奖励差异
            # ranked_reward_hoge = np.nan_to_num(ranked_reward_hoge, nan=0.0)
            # sampled_values = np.random.choice(len(ranked_reward_hoge), size= repeat_num, replace=True, p= ranked_reward_hoge/np.sum(ranked_reward_hoge))  # 基于概率分布采样索引
            # sample = sample[sampled_values, :]  # sample = pareto_sample[sampled_values, :]  # 执行重采样

            # --- ② Global Pareto Sampling: Resampling Based on the Worst-Case Score. by momst ---
            exp_scores = np.exp(robust_score * 5.0)
            exp_scores = np.nan_to_num(exp_scores, nan=0.0, posinf=0.0, neginf=0.0)
            if exp_scores.sum() == 0:
                probs = np.ones_like(exp_scores) / len(exp_scores)
            else:
                probs = exp_scores / exp_scores.sum()
            sampled_indices = np.random.choice(len(probs), size=repeat_num, replace=True, p=probs)
            sample = pareto_sample[sampled_indices, :]  # sample = sample[sampled_indices, :]

            all_generated_data = parse_trajectory_file(folder_path + '/trajectory.txt')
            if all_generated_data:
                analyze_protein_multi_rewards(
                    memory_bank=all_generated_data,
                    save_folder=folder_path,
                    reward_names=reward_metrics_name.split(','),
                    rewards_weights=rewards_weights
                )

            analyze_protein_multi_rewards(memory_bank, folder_path, reward_names=reward_metrics_name.split(','),
                                          rewards_weights=rewards_weights)

            # ① The sequence editing positions are fixed.
            loc_set = [random.sample(range(0,seq_len), edit_seqlength) for i in range(repeat_num)] # Location set
            length_edit = edit_seqlength

            for iii in range(repeat_num):
                sample[iii, loc_set[iii]] = mask


@torch.no_grad()
def generate_oaardm_reward_metrics_edit_initial(
        model,
        tokenizer,
        seq_len,
        reward_model,
        ori_pdb_file_path,
        batch,
        mask_for_loss,
        repeat_num,
        candidate,
        folder_path,
        device,
        ss_option = False,
        iteration = 30,
        edit_seqlength = 5,
        initial_sample = None  
):
    # Generate a random start string and convert to tokens
    mask = tokenizer.mask_id

    # Start from mask
    sample = initial_sample 
    sample = sample.to(device)
    # Unmask 1 loc at a time randomly
    loc = np.arange(seq_len)
    loc_set = [set(loc) for i in range(repeat_num)] # Location set   [[(0, 1, 2)], [(0, 1, 2, 3, 4, seq_len-1)]]
    with torch.no_grad():
        for ttt in tqdm(range(iteration)): 
            if ttt != 0:
                for count, kkk in enumerate(tqdm(range(length_edit))):
                    timestep = torch.tensor([0] * repeat_num)  # placeholder but not called in model
                    timestep = timestep.to(device)
                    prediction = model(sample, timestep)  # bs * seq len * dim
    
                    ### Make Candidates
                    next_candidate = []
                    pes_index_list = []
                    for jjj in range(candidate):
                        loc_list = np.random.randint(len(loc_set[0]), size=repeat_num)  # Choose random location
                        pes_index = [list(loc_set[i])[loc_list[i]] for i in range(repeat_num)]  # Which position
                        pes_index_list.append(pes_index)
                        p = torch.stack([prediction[i, pes_index[i], 0:20] for i in range(
                            repeat_num)])  # sample at location i (random), dont let it predict non-standard AA;
                        p = torch.nn.functional.softmax(p, dim=1)  # bs * 20
                        p_sample = torch.multinomial(p, num_samples=1)  # bs * 1
                        sample_fake = sample.clone()
                        for iii in range(repeat_num):
                            sample_fake[iii, pes_index[iii]] = p_sample.squeeze()[iii]
                        next_candidate.append(sample_fake.clone())  # Shape Next candidate: Rep * Batch size * Seq_length

                    ### reward & selection
                    reward_list = np.zeros((repeat_num, candidate))
                    for jjj in range(candidate):
                        prediction = model(next_candidate[jjj], timestep)
                        next_seq = next_candidate[jjj] * (next_candidate[jjj] != 28) + torch.argmax(prediction[:, :, 0:20],
                                                                                                    dim=2) * (
                                            next_candidate[jjj] == 28)
                    
                        
                        _, reward_hoge, _ = reward_model.reward_metrics(
                                    protein_name=batch,
                                    mask_for_loss=mask_for_loss,
                                    S_sp= next_seq,
                                    ori_pdb_file=ori_pdb_file_path,
                                )
                        reward_list[:, jjj] = reward_hoge
                    next_index = np.argmax(reward_list, 1)
                    
                    # update sample
                    next_candidate = torch.stack(next_candidate)
                    sample = torch.stack([next_candidate[next_index[i], i, :] for i in range(repeat_num)])
                    for jjj in range(repeat_num):
                        if loc_set[jjj]!=[]: 
                            loc_set[jjj].remove(pes_index_list[next_index[jjj]][jjj])
            
            _, reward_hoge, position_list  = reward_model.reward_metrics(
                        protein_name=batch,
                        mask_for_loss=mask_for_loss,
                        S_sp= sample,
                        ori_pdb_file=ori_pdb_file_path,
                    )
            
            print(reward_hoge)
            with open(folder_path +'/generate.txt', 'a') as f:
                f.write(', '.join(map(str, reward_hoge)) + '\n')
            
            if ttt == iteration-1:
                untokenized = [tokenizer.untokenize(s) for s in sample]
                return sample, untokenized
            
            reward_hoge = np.exp(np.array(reward_hoge)*5) ##np.exp(reward_hoge*4) #Transform
            sampled_values = np.random.choice([i for i in range(repeat_num)], size= repeat_num, p= reward_hoge/np.sum(reward_hoge))
            sample = sample[sampled_values, :]
           
   
            loc_set = [random.sample(range(0,seq_len), edit_seqlength) for i in range(repeat_num)] # Location set
            length_edit = edit_seqlength 
            for iii in range(repeat_num):
                sample[iii, loc_set[iii]] = mask


@torch.no_grad()
def generate_GA_reward_metrics(
        model,
        tokenizer,
        seq_len,
        reward_model,
        ori_pdb_file_path,
        batch,
        mask_for_loss,
        repeat_num,
        candidate,
        folder_path,
        device,
        ss_option = False,
        iteration = 30,
        edit_seqlength = 5,
        initial_sample = None  
):
    # Generate a random start string and convert to tokens
    mask = tokenizer.mask_id

    # Start from mask
    sample = initial_sample
    sample = sample.to(device)
    # Unmask 1 loc at a time randomly
    #loc = np.arange(seq_len)
    #loc_set = [set(loc) for i in range(repeat_num)] # Location set
    with torch.no_grad():
        for ttt in tqdm(range(iteration)): 
            if ttt != 0:
                for count, kkk in enumerate(tqdm(range(length_edit))):
                    timestep = torch.tensor([0] * repeat_num)  # placeholder but not called in model
                    timestep = timestep.to(device)
                    prediction = model(sample, timestep)  # bs * seq len * dim
                    loc_list = np.random.randint(len(loc_set[0]), size=repeat_num)  # Choose random location
                    pes_index = [list(loc_set[i])[loc_list[i]] for i in range(repeat_num)]  # Which position
                    p = torch.stack([prediction[i, pes_index[i], 0:20] for i in range(
                        repeat_num)])  # sample at location i (random), dont let it predict non-standard AA

                    p = torch.nn.functional.softmax(p, dim=1)  # bs * 20
                    p_sample = torch.multinomial(p, num_samples=1)  # bs * 1

                    for iii in range(repeat_num):
                        sample[iii, pes_index[iii]] = p_sample.squeeze()[iii]
                        loc_set[iii].remove(pes_index[iii])

            _, reward_hoge, position_list  = reward_model.reward_metrics(
                        protein_name=batch,
                        mask_for_loss=mask_for_loss,
                        S_sp= sample,
                        ori_pdb_file=ori_pdb_file_path,
                    )
            
            print(reward_hoge)
            with open(folder_path +'/generate.txt', 'a') as f:
                f.write(', '.join(map(str, reward_hoge)) + '\n')
            
            if ttt == iteration-1:
                untokenized = [tokenizer.untokenize(s) for s in sample]
                return sample, untokenized

            reward_hoge = np.exp(np.array(reward_hoge)*5) ##np.exp(reward_hoge*4) #Transform
            sampled_values = np.random.choice([i for i in range(repeat_num)], size= repeat_num, p= reward_hoge/np.sum(reward_hoge))

            sample = sample[sampled_values, :]

            loc_set = [random.sample(range(0,seq_len), edit_seqlength) for i in range(repeat_num)] # Location set
            length_edit = edit_seqlength
            for iii in range(repeat_num):
                sample[iii, loc_set[iii]] = mask


def generate_autoreg(model, tokenizer, samples=100, batch_size=1, max_seq_len=1024):
    # Generates 1 seq at a time, no batching, to make it easier to deal w variable seq lengths
    # Generates until max length or until stop token is predicted
    #model.eval().cuda()
    device = model.device()

    start = tokenizer.start_id
    stop = tokenizer.stop_id
    sample_out = []
    untokenized_out = []
    timestep = torch.tensor([0] * batch_size)  # placeholder but not called in model
    timestep = timestep.to(device)
    for s in tqdm(range(samples)):
        # Start from START token
        sample = (torch.zeros((1))+ start).unsqueeze(0) # add batch dim
        sample = sample.to(torch.long)
        sample = sample.to(device)
        # Iterate over each residue until desired length
        #max_loc = np.arange(max_seq_len)
        reach_stop=False # initialize
        with torch.no_grad():
            for i in range(max_seq_len):
                if reach_stop == False: # Add residues until it predicts STOP token or hits max seq len
                    prediction = model(sample, timestep) #, input_mask=input_mask.unsqueeze(-1)) #sample prediction given input
                    p = prediction[:, -1, :] # predict next token
                    p = torch.nn.functional.softmax(p, dim=1) # softmax over categorical probs
                    p_sample = torch.multinomial(p, num_samples=1)
                    sample = torch.cat((sample, p_sample), dim=1)
                    #print(tokenizer.untokenize(sample[0]))
                    #print(p_sample, stop)
                    if p_sample == stop:
                        reach_stop = True
                else:
                    break

        print("final seq", tokenizer.untokenize(sample[0,1:-1])) # dont save start/stop tokens
        untokenized = tokenizer.untokenize(sample[0,1:-1])
        sample_out.append(sample[0,1:-1])
        untokenized_out.append(untokenized)
    return sample_out, untokenized_out


def generate_d3pm(model, tokenizer, Q, Q_bar, timesteps, seq_len, batch_size=3, device='cuda'):
    """
    Generate a random start string from uniform dist and convert to predictions
    """
    #model.eval()
    #device = model.device()

    sample = torch.randint(0, tokenizer.K, (batch_size, seq_len))
    sample = sample.to(torch.long)
    sample = sample.to(device)
    Q = Q.to(device)
    Q_bar = Q_bar.to(device)

    timesteps = torch.linspace(timesteps-1,1,int((timesteps-1)/1), dtype=int) # iterate over reverse timesteps
    timesteps = timesteps.to(device)
    with torch.no_grad():
        for t in tqdm(timesteps):
            timesteps = torch.tensor([t] * batch_size)
            timesteps = timesteps.to(device)
            prediction = model(sample, timesteps)
            p = prediction[:, :, :tokenizer.K]  # p_theta_tilde (x_0_tilde | x_t) # Don't predict non-standard AAs
            p = torch.nn.functional.softmax(p, dim=-1)  # softmax over categorical probs
            p = p.to(torch.float64)
            x_tminus1 = sample.clone()
            for i, s in enumerate(sample):
                x_t_b = tokenizer.one_hot(s)
                A = torch.mm(x_t_b, torch.t(Q[t]))  # [P x K]
                Q_expand = Q_bar[t-1].unsqueeze(0).expand(A.shape[0], tokenizer.K, tokenizer.K)  # [ P x K x K]
                B_pred = torch.mul(p[i].unsqueeze(2), Q_expand)
                q_t = torch.mul(A.unsqueeze(1), B_pred)  # [ P x K x K ]
                p_theta_marg = torch.bmm(torch.transpose(q_t, 1,2),  p[i].unsqueeze(2)).squeeze()  # this marginalizes over dim=2
                p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)
                x_tminus1[i] = torch.multinomial(p_theta_marg, num_samples=1).squeeze()
                # On final timestep pick next best from standard AA
                if t == 1:
                     x_tminus1[i] = torch.multinomial(p_theta_marg[:, :tokenizer.K-6], num_samples=1).squeeze()
                # diff = torch.ne(s, x_tminus1[i])
                # if t % 100 == 0:
                #     print("time", t, diff.sum().item(), "mutations", tokenizer.untokenize(x_tminus1[i]), "sample", tokenizer.untokenize(s))
            sample = x_tminus1

    untokenized = [tokenizer.untokenize(s) for s in sample]
    print("final seq", untokenized)
    return sample, untokenized

def generate_random_seq(seq_len, train_prob_dist, tokenizer=Tokenizer()):
    """
    Generates a set of random sequences drawn from a train distribution
    """
    sample = torch.multinomial(torch.tensor(train_prob_dist), num_samples=seq_len, replacement=True)
    sample = sample.to(torch.long)
    return tokenizer.untokenize(sample)

def generate_valid_subset(data_valid, samples=20):
    sample = []
    for i in tqdm(range(samples)):
        r_idx = np.random.choice(len(data_valid))
        sequence = data_valid[r_idx][0]
        sample.append(sequence)
    print(sample)
    return sample


if __name__ == '__main__':
    main()