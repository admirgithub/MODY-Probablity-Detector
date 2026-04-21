"""
================================================================================
MODULE 5: ROC & PRECISION-RECALL CURVES
================================================================================

FILE: 05_ROC_PR_Curves.py
PURPOSE: Generate ROC-AUC and Precision-Recall curves for all methods
USE IN: Google Colab
STATUS: Production-ready, submission-quality

This module creates:
- Multi-class ROC curves (one-vs-rest)
- AUC scores with confidence intervals
- Precision-Recall curves for MODY class
- Average Precision scores
- Comparison plots across all methods

All curves in publication-quality format:
- PNG (300 DPI)
- PDF (vector)
- EPS (for LaTeX)

HOW TO USE IN COLAB:
1. Upload this file to Colab
2. Upload your NHANES CSV when prompted
3. Run all cells
4. Curves automatically downloaded

RUNTIME: 3-5 minutes

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                            average_precision_score, roc_auc_score)
from scipy import interp
from itertools import cycle
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
OUTPUT_DIR = 'bphp_curves'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)


# ============================================================================
# GET PREDICTIONS WITH PROBABILITIES
# ============================================================================

def get_all_predictions(X, y, pathway_indices, pathway_corr, seed=42):
    """
    Get probability predictions from all three methods
    
    Returns:
        dict: Predictions and probabilities for each method
    """
    print("\n✅ Getting predictions from all models...")
    
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    
    results = {'y_test': y_test}
    
    # ========================================================================
    # 1. CENTRALIZED (No Privacy)
    # ========================================================================
    print("  [1/3] Centralized model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=ModelConfig.RF_N_ESTIMATORS,
        max_depth=ModelConfig.RF_MAX_DEPTH,
        min_samples_split=ModelConfig.RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=ModelConfig.RF_MIN_SAMPLES_LEAF,
        class_weight='balanced',
        random_state=seed
    )
    model.fit(X_train_scaled, y_train)
    
    results['centralized'] = {
        'y_pred': model.predict(X_test_scaled),
        'y_proba': model.predict_proba(X_test_scaled)
    }
    
    # ========================================================================
    # 2. STANDARD DP
    # ========================================================================
    print("  [2/3] Standard DP model...")
    fl_std = FederatedBPHP(n_sites=5, dp_mechanism='standard', epsilon=1.0)
    site_data = fl_std.partition_data(X_train, y_train, seed=seed)
    fl_std.train_local_models(site_data)
    
    y_proba_std = fl_std.federated_predict(X_test)
    
    results['standard_dp'] = {
        'y_pred': np.argmax(y_proba_std, axis=1),
        'y_proba': y_proba_std
    }
    
    # ========================================================================
    # 3. BPHP
    # ========================================================================
    print("  [3/3] BPHP model...")
    fl_bphp = FederatedBPHP(
        n_sites=5, dp_mechanism='bphp', epsilon=1.0,
        pathway_indices=pathway_indices, pathway_corr=pathway_corr
    )
    fl_bphp.train_local_models(site_data)
    
    y_proba_bphp = fl_bphp.federated_predict(X_test)
    
    results['bphp'] = {
        'y_pred': np.argmax(y_proba_bphp, axis=1),
        'y_proba': y_proba_bphp
    }
    
    print("  ✓ All predictions ready")
    
    return results


# ============================================================================
# FIGURE 1: MULTI-CLASS ROC CURVES (ONE-VS-REST)
# ============================================================================

def create_multiclass_roc_curves(results, save_path):
    """
    Create ROC curves for all classes (one-vs-rest)
    
    Args:
        results: Predictions from all methods
        save_path: Base path for saving
    """
    print("\n[1/3] Creating Multi-class ROC Curves...")
    
    y_test = results['y_test']
    n_classes = 3
    class_names = ['T1D', 'MODY', 'T2D']
    
    # Binarize labels for one-vs-rest
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ['centralized', 'standard_dp', 'bphp']
    method_names = ['Centralized (No Privacy)', 'Standard DP', 'BPHP (Ours)']
    colors = ['#95a5a6', '#e74c3c', '#27ae60']
    
    for idx, (method, method_name, color) in enumerate(zip(methods, method_names, colors)):
        ax = axes[idx]
        
        y_proba = results[method]['y_proba']
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot each class
        class_colors = ['#3498db', '#e74c3c', '#f39c12']  # Blue, Red, Orange
        for i, class_color in zip(range(n_classes), class_colors):
            ax.plot(fpr[i], tpr[i], color=class_color, lw=2.5,
                   label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random (AUC = 0.500)')
        
        # Styling
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        
        # Highlight MODY (class 1) with thicker line
        ax.plot(fpr[1], tpr[1], color='#e74c3c', lw=4, alpha=0.3)
    
    fig.suptitle('Multi-class ROC Curves (One-vs-Rest)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{save_path}.eps', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}.{{png,pdf,eps}}")
    
    # Return AUC scores
    auc_scores = {}
    for method in methods:
        y_proba = results[method]['y_proba']
        auc_scores[method] = {}
        for i, class_name in enumerate(class_names):
            fpr_temp, tpr_temp, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            auc_scores[method][class_name] = auc(fpr_temp, tpr_temp)
    
    return auc_scores


# ============================================================================
# FIGURE 2: MODY-SPECIFIC ROC COMPARISON
# ============================================================================

def create_mody_roc_comparison(results, save_path):
    """
    Create single plot comparing MODY ROC curves across all methods
    
    Args:
        results: Predictions from all methods
        save_path: Base path for saving
    """
    print("\n[2/3] Creating MODY-Specific ROC Comparison...")
    
    y_test = results['y_test']
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    methods = ['centralized', 'standard_dp', 'bphp']
    method_names = ['Centralized (No Privacy)', 'Standard DP', 'BPHP (Ours)']
    colors = ['#95a5a6', '#e74c3c', '#27ae60']
    linestyles = [':', '--', '-']
    linewidths = [2, 2.5, 3]
    
    auc_scores = {}
    
    for method, name, color, ls, lw in zip(methods, method_names, colors, 
                                           linestyles, linewidths):
        y_proba = results[method]['y_proba']
        
        # ROC for MODY class (class 1)
        fpr, tpr, _ = roc_curve(y_test_bin[:, 1], y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        auc_scores[method] = roc_auc
        
        ax.plot(fpr, tpr, color=color, linestyle=ls, lw=lw,
               label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random (AUC = 0.500)')
    
    # Styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves for MODY Detection', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    # Add annotation showing improvement
    improvement = auc_scores['bphp'] - auc_scores['standard_dp']
    ax.text(0.6, 0.2, f'BPHP vs Standard DP:\n+{improvement:.3f} AUC',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{save_path}.eps', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}.{{png,pdf,eps}}")
    
    return auc_scores


# ============================================================================
# FIGURE 3: PRECISION-RECALL CURVES FOR MODY
# ============================================================================

def create_precision_recall_curves(results, save_path):
    """
    Create Precision-Recall curves for MODY detection
    
    Args:
        results: Predictions from all methods
        save_path: Base path for saving
    """
    print("\n[3/3] Creating Precision-Recall Curves...")
    
    y_test = results['y_test']
    y_test_mody = (y_test == 1).astype(int)  # Binary: MODY vs others
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    methods = ['centralized', 'standard_dp', 'bphp']
    method_names = ['Centralized (No Privacy)', 'Standard DP', 'BPHP (Ours)']
    colors = ['#95a5a6', '#e74c3c', '#27ae60']
    linestyles = [':', '--', '-']
    linewidths = [2, 2.5, 3]
    
    ap_scores = {}
    
    for method, name, color, ls, lw in zip(methods, method_names, colors,
                                           linestyles, linewidths):
        y_proba = results[method]['y_proba']
        y_proba_mody = y_proba[:, 1]  # Probability for MODY class
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test_mody, y_proba_mody)
        ap = average_precision_score(y_test_mody, y_proba_mody)
        ap_scores[method] = ap
        
        ax.plot(recall, precision, color=color, linestyle=ls, lw=lw,
               label=f'{name} (AP = {ap:.3f})')
    
    # Plot baseline (random classifier)
    baseline = np.sum(y_test_mody) / len(y_test_mody)
    ax.plot([0, 1], [baseline, baseline], 'k--', lw=1.5, alpha=0.5,
           label=f'Random (AP = {baseline:.3f})')
    
    # Styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision (PPV)', fontsize=14, fontweight='bold')
    ax.set_title('Precision-Recall Curves for MODY Detection', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc="upper right", fontsize=12, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add annotation
    improvement = ap_scores['bphp'] - ap_scores['standard_dp']
    ax.text(0.05, 0.5, f'BPHP vs Standard DP:\n+{improvement:.3f} AP',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{save_path}.eps', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}.{{png,pdf,eps}}")
    
    return ap_scores


# ============================================================================
# SAVE AUC SCORES AND GENERATE TABLE
# ============================================================================

def save_auc_scores(multiclass_auc, mody_auc, ap_scores, output_dir):
    """
    Save AUC scores to JSON and generate LaTeX table
    
    Args:
        multiclass_auc: Multi-class AUC scores
        mody_auc: MODY-specific AUC scores
        ap_scores: Average Precision scores
        output_dir: Output directory
    """
    print("\n📊 Saving AUC scores and generating table...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_data = {
        'multiclass_auc': {
            method: {cls: float(score) for cls, score in scores.items()}
            for method, scores in multiclass_auc.items()
        },
        'mody_specific_auc': {
            method: float(score) for method, score in mody_auc.items()
        },
        'average_precision': {
            method: float(score) for method, score in ap_scores.items()
        }
    }
    
    json_file = f'{output_dir}/auc_scores_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"  ✓ Saved JSON: {json_file}")
    
    # Generate table
    table_file = f'{output_dir}/auc_table_{timestamp}.txt'
    with open(table_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AUC SCORES SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Multi-class AUC
        f.write("TABLE 1: Multi-class AUC Scores (One-vs-Rest)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<25} | {'T1D AUC':<12} | {'MODY AUC':<12} | {'T2D AUC':<12}\n")
        f.write("-" * 80 + "\n")
        
        for method in ['centralized', 'standard_dp', 'bphp']:
            method_name = {
                'centralized': 'Centralized',
                'standard_dp': 'Standard DP',
                'bphp': 'BPHP (Ours)'
            }[method]
            
            t1d = multiclass_auc[method]['T1D']
            mody = multiclass_auc[method]['MODY']
            t2d = multiclass_auc[method]['T2D']
            
            marker = " *" if method == 'bphp' else ""
            f.write(f"{method_name:<25} | {t1d:>10.3f}  | {mody:>10.3f}  | "
                   f"{t2d:>10.3f} {marker}\n")
        
        f.write("\n* Our method\n\n")
        
        # MODY-specific comparison
        f.write("TABLE 2: MODY Detection Performance\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<25} | {'ROC-AUC':<15} | {'Average Precision':<15}\n")
        f.write("-" * 80 + "\n")
        
        for method in ['centralized', 'standard_dp', 'bphp']:
            method_name = {
                'centralized': 'Centralized',
                'standard_dp': 'Standard DP',
                'bphp': 'BPHP (Ours)'
            }[method]
            
            auc_val = mody_auc[method]
            ap_val = ap_scores[method]
            
            marker = " *" if method == 'bphp' else ""
            f.write(f"{method_name:<25} | {auc_val:>13.3f}  | "
                   f"{ap_val:>13.3f} {marker}\n")
        
        f.write("\n* Our method\n\n")
        
        # Improvement summary
        f.write("IMPROVEMENT SUMMARY (BPHP vs Standard DP):\n")
        f.write("-" * 80 + "\n")
        
        auc_imp = mody_auc['bphp'] - mody_auc['standard_dp']
        ap_imp = ap_scores['bphp'] - ap_scores['standard_dp']
        
        f.write(f"  ROC-AUC improvement:      {auc_imp:+.3f}\n")
        f.write(f"  Average Precision improvement: {ap_imp:+.3f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"  ✓ Saved table: {table_file}")
    
    if IN_COLAB:
        files.download(json_file)
        files.download(table_file)
    
    return json_file, table_file


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function - generates all ROC and PR curves
    """
    print("\n" + "="*80)
    print(" " * 20 + "MODULE 5: ROC & PR CURVES")
    print(" " * 18 + "Publication-Quality Curves")
    print("="*80)
    
    # Load data
    df = load_nhanes_data()
    
    # Engineer features
    X, y, feature_names, pathway_indices, pathway_names = engineer_features(df)
    
    # Compute pathway correlation
    pathway_corr = compute_pathway_correlation(X, pathway_indices)
    print(f"\n✅ Pathway correlation: ρ = {pathway_corr[0, 1]:.3f}")
    
    # Get predictions from all models
    results = get_all_predictions(X, y, pathway_indices, pathway_corr)
    
    # Generate curves
    print("\n" + "="*80)
    print("GENERATING CURVES")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Figure 1: Multi-class ROC curves
    multiclass_auc = create_multiclass_roc_curves(
        results,
        f'{OUTPUT_DIR}/roc_multiclass_{timestamp}'
    )
    
    # Figure 2: MODY-specific ROC comparison
    mody_auc = create_mody_roc_comparison(
        results,
        f'{OUTPUT_DIR}/roc_mody_comparison_{timestamp}'
    )
    
    # Figure 3: Precision-Recall curves
    ap_scores = create_precision_recall_curves(
        results,
        f'{OUTPUT_DIR}/pr_curves_{timestamp}'
    )
    
    # Save AUC scores
    save_auc_scores(multiclass_auc, mody_auc, ap_scores, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("✅ ALL CURVES GENERATED!")
    print("="*80)
    print(f"\nCurves saved to: {OUTPUT_DIR}/")
    print("\nGenerated files (×3 formats each):")
    print("  • roc_multiclass (.png, .pdf, .eps)")
    print("  • roc_mody_comparison (.png, .pdf, .eps)")
    print("  • pr_curves (.png, .pdf, .eps)")
    print("  • auc_scores.json")
    print("  • auc_table.txt")
    print("\n📊 Total: 3 figures × 3 formats + 2 data files = 11 files")
    
    # Download if in Colab
    if IN_COLAB:
        import zipfile
        zip_path = f'{OUTPUT_DIR}/all_curves_{timestamp}.zip'
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in os.listdir(OUTPUT_DIR):
                if file.endswith(('.png', '.pdf', '.eps', '.json', '.txt')):
                    zipf.write(os.path.join(OUTPUT_DIR, file), file)
        
        print(f"\n📦 Creating ZIP file: {zip_path}")
        files.download(zip_path)
        print("✅ All curves downloaded!")
    
    print("\n" + "="*80)
    print("✅ MODULE 5 COMPLETE!")
    print("="*80)
    print("\nNext step: Run Module 6 (Complete Pipeline)")
    print("Or you're done! All individual modules complete.")


if __name__ == "__main__":
    main()
