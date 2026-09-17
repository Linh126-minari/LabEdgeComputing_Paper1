#!/usr/bin/env python3
"""
Script to parse 3 training log files:
- train_lagrangianDQN.log
- train_dueling_dqn.log
- train_maddpg.log

and generate comprehensive, publication-quality comparison charts.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300

COLORS = {
    'LagrangianDQN': '#1f77b4',  # Vibrant Blue (Proposed)
    'DuelingDQN': '#ff7f0e',     # Orange (Baseline [A])
    'MADDPG': '#2ca02c',        # Green (Baseline MARL)
}

STYLES = {
    'LagrangianDQN': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2.2, 'marker': 'o', 'label': 'LagrangianDQN (Đề xuất)'},
    'DuelingDQN': {'color': '#ff7f0e', 'linestyle': '--', 'linewidth': 2.0, 'marker': 's', 'label': 'DuelingDQN (Baseline [A])'},
    'MADDPG': {'color': '#2ca02c', 'linestyle': '-.', 'linewidth': 2.0, 'marker': '^', 'label': 'MADDPG (Baseline MARL)'},
}

def parse_logs():
    files = {
        'LagrangianDQN': 'train_lagrangianDQN.log',
        'DuelingDQN': 'train_dueling_dqn.log',
        'MADDPG': 'train_maddpg.log'
    }
    
    train_data = {}
    eval_data = {}
    lagrangian_history = []
    
    for name, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Parse train lines: Episode    5 | Reward: -2518.2212 | Avg Loss:   1.100703 | Epsilon: 0.967000
        ep_pattern = r'Episode\s+(\d+)\s+\|\s+Reward:\s+([-+]?\d*\.?\d+)\s+\|\s+Avg Loss:\s+([-+]?\d*\.?\d+)\s+\|\s+Epsilon:\s+([-+]?\d*\.?\d+)'
        ep_matches = re.findall(ep_pattern, content)
        
        seen_eps = set()
        ep_records = []
        for ep, rew, loss, eps in ep_matches:
            ep = int(ep)
            if ep not in seen_eps:
                seen_eps.add(ep)
                ep_records.append({
                    'episode': ep,
                    'reward': float(rew),
                    'loss': float(loss),
                    'epsilon': float(eps),
                    'algorithm': name
                })
        train_data[name] = pd.DataFrame(ep_records)
        
        # Parse eval lines: Evaluating at episode 10... \n Eval Results: Reward=-2228.8861, Delay=0.423084s, Violation_Rate=6.40%
        eval_pattern = r'Evaluating at episode (\d+).*?Eval Results: Reward=([-+]?\d*\.?\d+), Delay=([-+]?\d*\.?\d+)s, Violation_Rate=([-+]?\d*\.?\d+)%'
        eval_matches = re.findall(eval_pattern, content, re.DOTALL)
        
        seen_evals = set()
        eval_records = []
        for ep, rew, delay, viol in eval_matches:
            ep = int(ep)
            if ep not in seen_evals:
                seen_evals.add(ep)
                eval_records.append({
                    'episode': ep,
                    'eval_reward': float(rew),
                    'delay': float(delay),
                    'violation_rate': float(viol),
                    'algorithm': name
                })
        eval_data[name] = pd.DataFrame(eval_records)
        
        # Parse Lagrangian specifics
        if name == 'LagrangianDQN':
            lag_pattern = r'\[Lagrangian\] ep=(\d+): viol_T=([-+]?\d*\.?\d+) \(δ=([-+]?\d*\.?\d+)\) → λ_T=([-+]?\d*\.?\d+); viol_E=([-+]?\d*\.?\d+) → λ_E=([-+]?\d*\.?\d+)'
            lag_matches = re.findall(lag_pattern, content)
            seen_lag = set()
            for ep, vt, dt, lt, ve, le in lag_matches:
                ep = int(ep)
                if ep not in seen_lag:
                    seen_lag.add(ep)
                    lagrangian_history.append({
                        'episode': ep,
                        'viol_T': float(vt),
                        'delta_T': float(dt),
                        'lambda_T': float(lt),
                        'viol_E': float(ve),
                        'lambda_E': float(le)
                    })
                    
    lag_df = pd.DataFrame(lagrangian_history)
    return train_data, eval_data, lag_df

def plot_dashboard(train_data, eval_data, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel (a): Training Reward
    ax = axes[0, 0]
    for name, df in train_data.items():
        st = STYLES[name]
        # Smooth with rolling mean
        smooth = df['reward'].rolling(window=5, min_periods=1).mean()
        ax.plot(df['episode'], smooth, label=st['label'], color=st['color'], linestyle=st['linestyle'], linewidth=st['linewidth'])
        ax.plot(df['episode'], df['reward'], color=st['color'], alpha=0.15, linewidth=1.0)
    ax.set_title('(a) Quá trình Hội tụ Phần thưởng Huấn luyện (Training Reward)', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (Điểm thưởng)')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Panel (b): Evaluation Reward
    ax = axes[0, 1]
    for name, df in eval_data.items():
        st = STYLES[name]
        ax.plot(df['episode'], df['eval_reward'], label=st['label'], color=st['color'], 
                linestyle=st['linestyle'], linewidth=st['linewidth'], marker=st['marker'], markevery=5, markersize=6)
    ax.set_title('(b) Phần thưởng Đánh giá (Evaluation Reward)', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Eval Reward')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Panel (c): Constraint Violation Rate
    ax = axes[1, 0]
    for name, df in eval_data.items():
        st = STYLES[name]
        ax.plot(df['episode'], df['violation_rate'], label=st['label'], color=st['color'], 
                linestyle=st['linestyle'], linewidth=st['linewidth'], marker=st['marker'], markevery=5, markersize=6)
    ax.axhline(y=5.0, color='red', linestyle=':', linewidth=2.0, label=r'Ngưỡng cho phép $\delta_T = 5\%$')
    ax.set_title(r'(c) Tỷ lệ Vi phạm Ràng buộc Deadline (Violation Rate %)', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Tỷ lệ vi phạm (%)')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Panel (d): Average Delay
    ax = axes[1, 1]
    for name, df in eval_data.items():
        st = STYLES[name]
        ax.plot(df['episode'], df['delay'], label=st['label'], color=st['color'], 
                linestyle=st['linestyle'], linewidth=st['linewidth'], marker=st['marker'], markevery=5, markersize=6)
    ax.axhline(y=1.0, color='darkred', linestyle='--', linewidth=1.5, label=r'Deadline ứng dụng $\tau_{app} = 1.0\text{s}$')
    ax.set_title(r'(d) Thời gian Trễ Thực thi Trung bình (Execution Delay)', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Thời gian trễ (giây)')
    ax.set_ylim(0.2, 0.6)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig_comparison_dashboard.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_reward_and_loss(train_data, eval_data, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Reward
    ax1 = axes[0]
    for name, df in train_data.items():
        st = STYLES[name]
        smooth = df['reward'].rolling(window=5, min_periods=1).mean()
        ax1.plot(df['episode'], smooth, label=st['label'], color=st['color'], linestyle=st['linestyle'], linewidth=st['linewidth'])
        ax1.fill_between(df['episode'], smooth - df['reward'].rolling(5).std().fillna(0),
                         smooth + df['reward'].rolling(5).std().fillna(0), color=st['color'], alpha=0.12)
    ax1.set_title('Đường cong Hội tụ Phần thưởng (Reward Convergence)', fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend(loc='center right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Training Loss
    ax2 = axes[1]
    for name, df in train_data.items():
        st = STYLES[name]
        smooth = df['loss'].rolling(window=5, min_periods=1).mean()
        ax2.plot(df['episode'], smooth, label=st['label'], color=st['color'], linestyle=st['linestyle'], linewidth=st['linewidth'])
    ax2.set_title('Tổn thất Huấn luyện Trung bình (Average TD Loss)', fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Smooth L1 Loss')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig_reward_and_loss.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_lagrangian_analysis(lag_df, output_dir):
    if lag_df.empty:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # Multipliers
    ax1 = axes[0]
    ax1.plot(lag_df['episode'], lag_df['lambda_T'], label=r'Nhân tử trễ hạn $\lambda_T$', color='#1f77b4', linewidth=2.5, marker='o', markersize=5)
    ax1.plot(lag_df['episode'], lag_df['lambda_E'], label=r'Nhân tử năng lượng $\lambda_E$', color='#d62728', linewidth=2.5, marker='s', markersize=5)
    ax1.set_title(r'(a) Tiến trình Điều chỉnh Nhân tử Lagrangian ($\lambda_T, \lambda_E$)', fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Giá trị Nhân tử (Multiplier Value)')
    ax1.legend(loc='center left', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Violation tracking
    ax2 = axes[1]
    ax2.plot(lag_df['episode'], lag_df['viol_T'] * 100, label=r'Tỷ lệ vi phạm trễ $viol_T$', color='#2ca02c', linewidth=2.2, marker='^', markersize=5)
    ax2.axhline(y=lag_df['delta_T'].iloc[0] * 100, color='red', linestyle='--', linewidth=2.0, label=r'Mục tiêu cho phép $\delta_T = 5\%$')
    ax2.set_title(r'(b) Tỷ lệ Vi phạm Ràng buộc Thực tế vs Ngưỡng $\delta_T$', fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Tỷ lệ vi phạm (%)')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig_lagrangian_multipliers.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_final_performance_bars(eval_data, output_dir):
    # Take last 10 evals (episodes 410 -> 500)
    stats = []
    for name, df in eval_data.items():
        last_evals = df.iloc[-10:]
        stats.append({
            'Algorithm': name,
            'Label': STYLES[name]['label'].split(' (')[0],
            'Mean_Reward': last_evals['eval_reward'].mean(),
            'Std_Reward': last_evals['eval_reward'].std(),
            'Mean_Delay': last_evals['delay'].mean(),
            'Std_Delay': last_evals['delay'].std(),
            'Mean_Violation': last_evals['violation_rate'].mean(),
            'Std_Violation': last_evals['violation_rate'].std(),
        })
    df_stats = pd.DataFrame(stats)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Bar 1: Reward
    ax1 = axes[0]
    bars1 = ax1.bar(df_stats['Label'], df_stats['Mean_Reward'], yerr=df_stats['Std_Reward'], 
                    capsize=6, color=[COLORS[a] for a in df_stats['Algorithm']], edgecolor='black', alpha=0.85)
    ax1.set_title('(a) Thưởng Trung bình (Reward)', fontweight='bold')
    ax1.set_ylabel('Mean Reward')
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval/2.0, f'{yval:.1f}', ha='center', va='center', color='white', fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Bar 2: Delay
    ax2 = axes[1]
    bars2 = ax2.bar(df_stats['Label'], df_stats['Mean_Delay'], yerr=df_stats['Std_Delay'], 
                    capsize=6, color=[COLORS[a] for a in df_stats['Algorithm']], edgecolor='black', alpha=0.85)
    ax2.set_title('(b) Trễ Xử lý Trung bình (Delay)', fontweight='bold')
    ax2.set_ylabel('Thời gian trễ (s)')
    ax2.set_ylim(0, 0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f'{yval:.3f}s', ha='center', va='bottom', fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Bar 3: Violation Rate
    ax3 = axes[2]
    bars3 = ax3.bar(df_stats['Label'], df_stats['Mean_Violation'], yerr=df_stats['Std_Violation'], 
                    capsize=6, color=[COLORS[a] for a in df_stats['Algorithm']], edgecolor='black', alpha=0.85)
    ax3.axhline(y=5.0, color='red', linestyle='--', linewidth=1.8, label=r'Ngưỡng $\delta_T = 5\%$')
    ax3.set_title('(c) Tỷ lệ Vi phạm Deadline (%)', fontweight='bold')
    ax3.set_ylabel('Tỷ lệ vi phạm (%)')
    ax3.set_ylim(0, 8.0)
    ax3.legend(loc='upper right', frameon=True)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')
    ax3.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig_final_performance_bars.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    # Also save table to CSV
    csv_path = os.path.join(output_dir, 'summary_comparison_table.csv')
    df_stats.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")

def main():
    output_dir = 'outputs/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Parsing logs...")
    train_data, eval_data, lag_df = parse_logs()
    
    print("Generating figures...")
    plot_dashboard(train_data, eval_data, output_dir)
    plot_reward_and_loss(train_data, eval_data, output_dir)
    plot_lagrangian_analysis(lag_df, output_dir)
    plot_final_performance_bars(eval_data, output_dir)
    print("All comparison plots generated successfully in:", output_dir)

if __name__ == '__main__':
    main()
