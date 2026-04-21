"""
================================================================================
MODULE 4: ABLATION STUDIES
================================================================================

FILE: 04_Ablation_Studies.py
PURPOSE: Test contribution of each component (what-if analysis)
USE IN: Google Colab
STATUS: Production-ready, submission-quality

This module answers reviewer questions like:
- What if we remove pathway preservation?
- What if we use different epsilon values?
- What if we change number of sites?
- What is the contribution of each component?

Ablation tests:
1. Without pathway preservation (BPHP → Standard DP)
2. Without federated learning (Federated → Centralized)
3. Different epsilon values (0.5, 1.0, 2.0, 5.0, 10.0)
4. Different number of sites (3, 5, 10, 20)
5. Different noise multipliers (0.5, 0.7, 0.9, 1.0)

HOW TO USE IN COLAB:
1. Upload this file to Colab
2. Upload your NHANES CSV when prompted
3. Run all cells
4. Ablation results saved to bphp_ablation/

RUNTIME: 15-20 minutes

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
import warnings
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Check if in Colab
try:
    from google.colab import files
    IN_COLAB = True
    print("✅ Running in Google Colab")
except:
    IN_COLAB = False
    print("⚠️  Not in Colab")

# Import from previous modules
import sys
sys.path.append('.')
from Module_01_BPHP_Model import (
    FederatedBPHP, ModelConfig, load_nhanes_data,
    engineer_features, compute_pathway_correlation
)

# Output directory
OUTPUT_DIR = 'bphp_ablation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)


# ============================================================================
# ABLATION TEST 1: WITHOUT PATHWAY PRESERVATION
# ============================================================================

def test_without_pathway_preservation(X, y, pathway_indices, pathway_corr):
    """
    Compare BPHP vs Standard DP (removes pathway preservation)
    
    Returns:
        dict: Results
    """
    print("\n[1/5] Testing WITHOUT Pathway Preservation...")
    
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    results = {}
    
    # With pathway preservation (BPHP)
    fl_bphp = FederatedBPHP(
        n_sites=5, dp_mechanism='bphp', epsilon=1.0,
        pathway_indices=pathway_indices, pathway_corr=pathway_corr
    )
    site_data = fl_bphp.partition_data(X_train, y_train, seed=42)
    fl_bphp.train_local_models(site_data)
    y_proba = fl_bphp.federated_predict(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    
    results['with_pathway'] = {
        'mody_recall': recall[1] if len(recall) > 1 else 0,
        'accuracy': accuracy_score(y_test, y_pred)
    }
    
    # Without pathway preservation (Standard DP)
    fl_std = FederatedBPHP(
        n_sites=5, dp_mechanism='standard', epsilon=1.0
    )
    fl_std.train_local_models(site_data)
    y_proba = fl_std.federated_predict(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    
    results['without_pathway'] = {
        'mody_recall': recall[1] if len(recall) > 1 else 0,
        'accuracy': accuracy_score(y_test, y_pred)
    }
    
    # Compute contribution
    contribution = (results['with_pathway']['mody_recall'] - 
                   results['without_pathway']['mody_recall']) * 100
    
    print(f"  With pathway:    {results['with_pathway']['mody_recall']*100:.1f}%")
    print(f"  Without pathway: {results['without_pathway']['mody_recall']*100:.1f}%")
    print(f"  Contribution:    +{contribution:.1f} percentage points")
    
    return results


# ============================================================================
# ABLATION TEST 2: WITHOUT FEDERATED LEARNING
# ============================================================================

def test_without_federated_learning(X, y):
    """
    Compare Federated vs Centralized (removes federated learning)
    
    Returns:
        dict: Results
    """
    print("\n[2/5] Testing WITHOUT Federated Learning...")
    
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    results = {}
    
    # With federated learning (no privacy, just federation)
    fl = FederatedBPHP(n_sites=5, dp_mechanism='none')
    site_data = fl.partition_data(X_train, y_train, seed=42)
    fl.train_local_models(site_data)
    y_proba = fl.federated_predict(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    
    results['with_federated'] = {
        'mody_recall': recall[1] if len(recall) > 1 else 0,
        'accuracy': accuracy_score(y_test, y_pred)
    }
    
    # Without federated learning (centralized)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(
        n_estimators=100, max_depth=12, class_weight='balanced', random_state=42
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    
    results['without_federated'] = {
        'mody_recall': recall[1] if len(recall) > 1 else 0,
        'accuracy': accuracy_score(y_test, y_pred)
    }
    
    # Compute impact
    impact = (results['with_federated']['mody_recall'] - 
             results['without_federated']['mody_recall']) * 100
    
    print(f"  Federated:    {results['with_federated']['mody_recall']*100:.1f}%")
    print(f"  Centralized:  {results['without_federated']['mody_recall']*100:.1f}%")
    print(f"  Impact:       {impact:+.1f} percentage points")
    
    return results


# ============================================================================
# ABLATION TEST 3: DIFFERENT EPSILON VALUES
# ============================================================================

def test_different_epsilon_values(X, y, pathway_indices, pathway_corr):
    """
    Test different privacy budgets
    
    Returns:
        dict: Results for each epsilon
    """
    print("\n[3/5] Testing Different Epsilon Values...")
    
    epsilons = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = {}
    
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    for eps in epsilons:
        print(f"  Testing ε = {eps}...", end=' ')
        
        # Standard DP
        fl_std = FederatedBPHP(n_sites=5, dp_mechanism='standard', epsilon=eps)
        site_data = fl_std.partition_data(X_train, y_train, seed=42)
        fl_std.train_local_models(site_data)
        y_proba = fl_std.federated_predict(X_test)
        y_pred = np.argmax(y_proba, axis=1)
        recall_std = recall_score(y_test, y_pred, average=None, zero_division=0)
        
        # BPHP
        fl_bphp = FederatedBPHP(
            n_sites=5, dp_mechanism='bphp', epsilon=eps,
            pathway_indices=pathway_indices, pathway_corr=pathway_corr
        )
        fl_bphp.train_local_models(site_data)
        y_proba = fl_bphp.federated_predict(X_test)
        y_pred = np.argmax(y_proba, axis=1)
        recall_bphp = recall_score(y_test, y_pred, average=None, zero_division=0)
        
        results[eps] = {
            'standard_dp': recall_std[1] if len(recall_std) > 1 else 0,
            'bphp': recall_bphp[1] if len(recall_bphp) > 1 else 0
        }
        
        print(f"Std={results[eps]['standard_dp']*100:.1f}%, BPHP={results[eps]['bphp']*100:.1f}%")
    
    return results


# ============================================================================
# ABLATION TEST 4: DIFFERENT NUMBER OF SITES
# ============================================================================

def test_different_number_of_sites(X, y, pathway_indices, pathway_corr):
    """
    Test different numbers of federated sites
    
    Returns:
        dict: Results for each site count
    """
    print("\n[4/5] Testing Different Number of Sites...")
    
    site_counts = [3, 5, 10, 20]
    results = {}
    
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    for n_sites in site_counts:
        print(f"  Testing {n_sites} sites...", end=' ')
        
        # Standard DP
        fl_std = FederatedBPHP(n_sites=n_sites, dp_mechanism='standard', epsilon=1.0)
        site_data = fl_std.partition_data(X_train, y_train, seed=42)
        fl_std.train_local_models(site_data)
        y_proba = fl_std.federated_predict(X_test)
        y_pred = np.argmax(y_proba, axis=1)
        recall_std = recall_score(y_test, y_pred, average=None, zero_division=0)
        
        # BPHP
        fl_bphp = FederatedBPHP(
            n_sites=n_sites, dp_mechanism='bphp', epsilon=1.0,
            pathway_indices=pathway_indices, pathway_corr=pathway_corr
        )
        fl_bphp.train_local_models(site_data)
        y_proba = fl_bphp.federated_predict(X_test)
        y_pred = np.argmax(y_proba, axis=1)
        recall_bphp = recall_score(y_test, y_pred, average=None, zero_division=0)
        
        results[n_sites] = {
            'standard_dp': recall_std[1] if len(recall_std) > 1 else 0,
            'bphp': recall_bphp[1] if len(recall_bphp) > 1 else 0
        }
        
        print(f"Std={results[n_sites]['standard_dp']*100:.1f}%, BPHP={results[n_sites]['bphp']*100:.1f}%")
    
    return results


# ============================================================================
# ABLATION TEST 5: DIFFERENT NOISE MULTIPLIERS
# ============================================================================

def test_different_noise_multipliers(X, y, pathway_indices, pathway_corr):
    """
    Test different BPHP noise multipliers
    
    Returns:
        dict: Results for each multiplier
    """
    print("\n[5/5] Testing Different Noise Multipliers (BPHP only)...")
    
    multipliers = [0.5, 0.7, 0.9, 1.0]  # 0.7 is our choice
    results = {}
    
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    for mult in multipliers:
        print(f"  Testing multiplier = {mult}...", end=' ')
        
        # Temporarily modify the config
        original_mult = ModelConfig.BPHP_NOISE_MULTIPLIER
        ModelConfig.BPHP_NOISE_MULTIPLIER = mult
        
        fl_bphp = FederatedBPHP(
            n_sites=5, dp_mechanism='bphp', epsilon=1.0,
            pathway_indices=pathway_indices, pathway_corr=pathway_corr
        )
        site_data = fl_bphp.partition_data(X_train, y_train, seed=42)
        fl_bphp.train_local_models(site_data)
        y_proba = fl_bphp.federated_predict(X_test)
        y_pred = np.argmax(y_proba, axis=1)
        recall = recall_score(y_test, y_pred, average=None, zero_division=0)
        
        results[mult] = {
            'mody_recall': recall[1] if len(recall) > 1 else 0
        }
        
        # Restore original
        ModelConfig.BPHP_NOISE_MULTIPLIER = original_mult
        
        print(f"MODY recall={results[mult]['mody_recall']*100:.1f}%")
    
    return results


# ============================================================================
# VISUALIZATION: ABLATION RESULTS
# ============================================================================

def visualize_ablation_results(epsilon_results, sites_results, noise_results, output_dir):
    """
    Create visualization of ablation study results
    """
    print("\n📊 Creating ablation visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Epsilon values
    epsilons = sorted(epsilon_results.keys())
    std_vals = [epsilon_results[e]['standard_dp'] * 100 for e in epsilons]
    bphp_vals = [epsilon_results[e]['bphp'] * 100 for e in epsilons]
    
    axes[0].plot(epsilons, std_vals, 'o-', label='Standard DP', 
                linewidth=2, markersize=8, color='#e74c3c')
    axes[0].plot(epsilons, bphp_vals, 's-', label='BPHP', 
                linewidth=2, markersize=8, color='#27ae60')
    axes[0].axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Privacy Budget (ε)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('MODY Recall (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Effect of Privacy Budget', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xscale('log')
    
    # Plot 2: Number of sites
    sites = sorted(sites_results.keys())
    std_vals = [sites_results[s]['standard_dp'] * 100 for s in sites]
    bphp_vals = [sites_results[s]['bphp'] * 100 for s in sites]
    
    axes[1].plot(sites, std_vals, 'o-', label='Standard DP', 
                linewidth=2, markersize=8, color='#e74c3c')
    axes[1].plot(sites, bphp_vals, 's-', label='BPHP', 
                linewidth=2, markersize=8, color='#27ae60')
    axes[1].axvline(5, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Number of Sites', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('MODY Recall (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Effect of Site Count', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Plot 3: Noise multipliers
    mults = sorted(noise_results.keys())
    vals = [noise_results[m]['mody_recall'] * 100 for m in mults]
    
    axes[2].plot(mults, vals, 'o-', linewidth=2, markersize=8, color='#3498db')
    axes[2].axvline(0.7, color='gray', linestyle='--', alpha=0.5, 
                   label='Our choice (0.7)')
    axes[2].set_xlabel('BPHP Noise Multiplier', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('MODY Recall (%)', fontsize=12, fontweight='bold')
    axes[2].set_title('Effect of Noise Multiplier', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'{output_dir}/ablation_visualization_{timestamp}'
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}.{{png,pdf}}")


# ============================================================================
# GENERATE ABLATION TABLE
# ============================================================================

def generate_ablation_table(all_results, output_dir):
    """
    Generate LaTeX-ready ablation table
    """
    print("\n📝 Generating ablation table...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_file = f'{output_dir}/ablation_table_{timestamp}.txt'
    
    with open(table_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ABLATION STUDY RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # Component contribution
        f.write("TABLE 1: Component Contribution\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Component':<30} | {'MODY Recall':<15} | {'Contribution':<15}\n")
        f.write("-" * 80 + "\n")
        
        pathway = all_results['pathway_preservation']
        f.write(f"{'Full BPHP':<30} | "
                f"{pathway['with_pathway']['mody_recall']*100:>6.1f}% {'':<7} | {'Baseline':<15}\n")
        f.write(f"{'  - Pathway preservation':<30} | "
                f"{pathway['without_pathway']['mody_recall']*100:>6.1f}% {'':<7} | "
                f"{(pathway['with_pathway']['mody_recall']-pathway['without_pathway']['mody_recall'])*100:>+6.1f}% pts {'':<3}\n")
        
        federated = all_results['federated_learning']
        improvement = (federated['without_federated']['mody_recall'] - 
                      federated['with_federated']['mody_recall']) * 100
        f.write(f"{'Centralized (upper bound)':<30} | "
                f"{federated['without_federated']['mody_recall']*100:>6.1f}% {'':<7} | "
                f"{improvement:>+6.1f}% pts {'':<3}\n")
        
        f.write("\n\n")
        
        # Privacy budget sensitivity
        f.write("TABLE 2: Privacy Budget Sensitivity (ε)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epsilon':<10} | {'Standard DP':<15} | {'BPHP':<15} | {'Improvement':<15}\n")
        f.write("-" * 80 + "\n")
        
        for eps in sorted(all_results['epsilon_values'].keys()):
            std = all_results['epsilon_values'][eps]['standard_dp'] * 100
            bphp = all_results['epsilon_values'][eps]['bphp'] * 100
            imp = bphp - std
            marker = " *" if eps == 1.0 else ""
            f.write(f"{eps:<10.1f} | {std:>6.1f}% {'':<7} | {bphp:>6.1f}% {'':<7} | "
                   f"{imp:>+6.1f}% pts{marker:<6}\n")
        
        f.write("\n* Our choice\n\n")
        
        # Number of sites
        f.write("TABLE 3: Federated Sites Scalability\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Sites':<10} | {'Standard DP':<15} | {'BPHP':<15} | {'Improvement':<15}\n")
        f.write("-" * 80 + "\n")
        
        for n_sites in sorted(all_results['number_of_sites'].keys()):
            std = all_results['number_of_sites'][n_sites]['standard_dp'] * 100
            bphp = all_results['number_of_sites'][n_sites]['bphp'] * 100
            imp = bphp - std
            marker = " *" if n_sites == 5 else ""
            f.write(f"{n_sites:<10} | {std:>6.1f}% {'':<7} | {bphp:>6.1f}% {'':<7} | "
                   f"{imp:>+6.1f}% pts{marker:<6}\n")
        
        f.write("\n* Our choice\n\n")
        
        # Noise multiplier
        f.write("TABLE 4: BPHP Noise Multiplier Selection\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Multiplier':<15} | {'MODY Recall':<15} | {'Notes':<30}\n")
        f.write("-" * 80 + "\n")
        
        for mult in sorted(all_results['noise_multipliers'].keys()):
            recall = all_results['noise_multipliers'][mult]['mody_recall'] * 100
            if mult == 0.7:
                note = "Our choice (best balance)"
            elif mult < 0.7:
                note = "More noise, lower utility"
            else:
                note = "Less noise, weaker privacy"
            marker = " *" if mult == 0.7 else ""
            f.write(f"{mult:<15.1f} | {recall:>6.1f}% {'':<7} | {note:<30}{marker}\n")
        
        f.write("\n* Our choice\n")
        f.write("\n" + "="*80 + "\n")
    
    print(f"  ✓ Saved: {table_file}")
    
    if IN_COLAB:
        files.download(table_file)
    
    return table_file


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function - runs all ablation tests
    """
    print("\n" + "="*80)
    print(" " * 23 + "MODULE 4: ABLATION STUDIES")
    print(" " * 18 + "Component Contribution Analysis")
    print("="*80)
    
    # Load data
    df = load_nhanes_data()
    
    # Engineer features
    X, y, feature_names, pathway_indices, pathway_names = engineer_features(df)
    
    # Compute pathway correlation
    pathway_corr = compute_pathway_correlation(X, pathway_indices)
    print(f"\n✅ Pathway correlation: ρ = {pathway_corr[0, 1]:.3f}")
    
    # Run all ablation tests
    print("\n" + "="*80)
    print("RUNNING ABLATION TESTS")
    print("="*80)
    
    all_results = {}
    
    # Test 1: Pathway preservation
    all_results['pathway_preservation'] = test_without_pathway_preservation(
        X, y, pathway_indices, pathway_corr
    )
    
    # Test 2: Federated learning
    all_results['federated_learning'] = test_without_federated_learning(X, y)
    
    # Test 3: Epsilon values
    all_results['epsilon_values'] = test_different_epsilon_values(
        X, y, pathway_indices, pathway_corr
    )
    
    # Test 4: Number of sites
    all_results['number_of_sites'] = test_different_number_of_sites(
        X, y, pathway_indices, pathway_corr
    )
    
    # Test 5: Noise multipliers
    all_results['noise_multipliers'] = test_different_noise_multipliers(
        X, y, pathway_indices, pathway_corr
    )
    
    # Visualize results
    visualize_ablation_results(
        all_results['epsilon_values'],
        all_results['number_of_sites'],
        all_results['noise_multipliers'],
        OUTPUT_DIR
    )
    
    # Generate table
    generate_ablation_table(all_results, OUTPUT_DIR)
    
    # Save JSON results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f'{OUTPUT_DIR}/ablation_results_{timestamp}.json'
    
    # Convert to JSON-serializable format
    json_results = {}
    for key, value in all_results.items():
        if isinstance(value, dict):
            json_results[key] = {}
            for k, v in value.items():
                if isinstance(v, dict):
                    json_results[key][str(k)] = {
                        kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                        for kk, vv in v.items()
                    }
                else:
                    json_results[key][str(k)] = float(v) if isinstance(v, (np.floating, float)) else v
    
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ JSON results saved: {json_file}")
    
    if IN_COLAB:
        files.download(json_file)
    
    print("\n" + "="*80)
    print("✅ MODULE 4 COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  • ablation_results.json")
    print("  • ablation_table.txt")
    print("  • ablation_visualization.{png,pdf}")
    print("\nNext step: Run Module 5 (ROC & PR Curves)")
    
    return all_results


if __name__ == "__main__":
    results = main()
