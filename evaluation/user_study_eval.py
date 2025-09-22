"""
CSV Model Evaluation Analysis Pipeline

Processes model evaluation data with statistical analysis and publication-ready visualizations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_TYPES = ["orange-model", "blue-model", "green-model", "purple-model"]
CRITERIA = ["Clarity of Steps", "Ease of Following", "Confidence"]

MODEL_LABELS = {
    'orange-model': 'OSS',
    'blue-model': 'DAPO', 
    'green-model': 'QwQ+DAPO/OSS',
    'purple-model': 'QwQ+OSS/DAPO',
    'best-overall': 'Best Overall'
}

MODEL_COLORS = {
    'OSS': '#FFB366',           # Orange
    'DAPO': '#6BB6FF',          # Blue  
    'QwQ+DAPO/OSS': '#66D9A3',  # Green
    'QwQ+OSS/DAPO': '#B366FF',  # Purple
    'Best Overall': '#FFD666',   # Yellow
    'default': '#FFACAC'         # Pink fallback
}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def process_csv_complete_with_diagnostics_and_drop(
    file_path, 
    output_dir="results", 
    visualizations_dir="visualizations",
    drop_incomplete=False, 
    run_ttests=True, 
    bonferroni_correction=True, 
    create_plots=True
):
    """
    Process model evaluation CSV with statistical analysis and visualizations.
    
    Returns: (expanded_dataframe, clean_dataframe, ttest_results)
    """
    # Setup
    action = "Dropping" if drop_incomplete else "Keeping"
    print(f"=== Model Evaluation Pipeline + {action} Incomplete Rows ===")
    print(f"📁 Output: {output_dir}" + (f" | Plots: {visualizations_dir}" if create_plots else ""))
    
    _create_directories(output_dir, visualizations_dir if create_plots else None)
    
    # Process data
    df_clean = _process_csv(file_path, drop_incomplete)
    df_expanded = _expand_data(df_clean)
    
    # Analysis and visualization
    ttest_results = _analyze_data(df_expanded, bonferroni_correction) if run_ttests and len(df_expanded) > 0 else None
    
    if create_plots and len(df_expanded) > 0:
        _create_plots(df_expanded, visualizations_dir, drop_incomplete)
    
    _save_data(df_clean, df_expanded, ttest_results, output_dir, drop_incomplete)
    
    print("\n✅ Analysis complete!")
    return df_expanded, df_clean, ttest_results

# =============================================================================
# DATA PROCESSING
# =============================================================================

def _process_csv(file_path, drop_incomplete):
    """Load and process CSV file."""
    print(f"📖 Loading: {file_path}")
    
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        raise Exception(f"Error loading CSV: {e}")
    
    # Clean headers
    df = df.drop(df.index[0]).reset_index(drop=True)
    df.columns = df.iloc[0].tolist()
    df = df.drop(df.index[:2]).reset_index(drop=True)
    
    # Filter relevant columns
    target_cols = MODEL_TYPES + ["best-overall"]
    relevant_cols = [col for col in df.columns if any(target in str(col) for target in target_cols)]
    df = df[relevant_cols]
    
    # Handle incomplete rows
    complete_rows = _find_complete_rows(df)
    
    if drop_incomplete and len(complete_rows) < len(df):
        print(f"🗑️ Dropping {len(df) - len(complete_rows)} incomplete rows")
        df = df.iloc[complete_rows].reset_index(drop=True)
    else:
        if len(complete_rows) < len(df):
            print(f"⚠️ Keeping {len(df) - len(complete_rows)} incomplete rows")
    
    print(f"📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def _find_complete_rows(df):
    """Find rows with complete data (5 values per model+criterion)."""
    complete_rows = []
    all_types = MODEL_TYPES + ["best-overall"]
    
    for idx in df.index:
        is_complete = True
        for model in all_types:
            for criterion in CRITERIA:
                cols = [c for c in df.columns if model in str(c) and criterion in str(c)]
                if cols:
                    valid_count = sum(1 for col in cols if _is_valid_number(df.at[idx, col]))
                    if valid_count != 5:
                        is_complete = False
                        break
            if not is_complete:
                break
        if is_complete:
            complete_rows.append(idx)
    
    return complete_rows

def _is_valid_number(value):
    """Check if value is a valid number."""
    if pd.notna(value) and str(value).strip():
        try:
            float(str(value).strip())
            return True
        except (ValueError, TypeError):
            pass
    return False

def _expand_data(df):
    """Transform data from wide to long format."""
    print("🔄 Expanding to long format...")
    
    data = []
    
    # Regular criteria
    for model in MODEL_TYPES:
        for criterion in CRITERIA:
            cols = [c for c in df.columns if model in str(c) and criterion in str(c)]
            _extract_data(df, cols, model, criterion, data)
    
    # Best-overall ratings
    best_cols = [c for c in df.columns if 'best-overall' in str(c).lower()]
    for col in best_cols:
        model = _infer_model(col)
        _extract_data(df, [col], model, 'best-overall', data, 'best_overall')
    
    df_expanded = pd.DataFrame(data)
    if len(df_expanded) > 0:
        df_expanded['model_label'] = df_expanded['model'].map(MODEL_LABELS)
        print(f"📈 Created {len(df_expanded)} data points")
    
    return df_expanded

def _extract_data(df, cols, model, criterion, data, group_prefix=None):
    """Extract numeric data from columns."""
    group_name = f"{group_prefix}_{model}" if group_prefix else f"{model}_{criterion.replace(' ', '_')}"
    
    for col in cols:
        for idx in df.index:
            value = _get_float(df.at[idx, col])
            if value is not None:
                data.append({
                    'original_row': idx, 'original_col': col, 'group': group_name,
                    'model': model, 'criterion': criterion, 'value': value
                })

def _get_float(value):
    """Convert value to float safely."""
    if pd.notna(value) and str(value).strip():
        try:
            return float(str(value).strip())
        except (ValueError, TypeError):
            pass
    return None

def _infer_model(col):
    """Infer model type from column name."""
    col_lower = str(col).lower()
    for model in MODEL_TYPES:
        if model.split('-')[0] in col_lower:
            return model
    return 'best-overall'

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def _analyze_data(df_expanded, bonferroni_correction):
    """Conduct pairwise t-tests with optional Bonferroni correction."""
    print("\n=== Statistical Analysis ===")
    
    results = []
    
    # Test regular criteria
    regular_data = df_expanded[df_expanded['criterion'] != 'best-overall']
    if len(regular_data) > 0:
        criteria = regular_data['criterion'].unique()
        models = regular_data['model'].unique()
        print(f"🧪 Testing {len(criteria)} criteria across {len(models)} models")
        
        for criterion in criteria:
            print(f"\n--- {criterion} ---")
            crit_data = regular_data[regular_data['criterion'] == criterion]
            for m1, m2 in combinations(models, 2):
                result = _run_ttest(crit_data, m1, m2, criterion)
                if result:
                    results.append(result)
    
    # Test best-overall
    best_data = df_expanded[df_expanded['criterion'] == 'best-overall']
    if len(best_data) > 0:
        print("\n--- Best Overall Rankings ---")
        models = best_data['model'].unique()
        for m1, m2 in combinations(models, 2):
            result = _run_ttest(best_data, m1, m2, 'best-overall')
            if result:
                results.append(result)
    
    return _process_test_results(results, bonferroni_correction)

def _run_ttest(data, model1, model2, criterion):
    """Perform t-test between two models."""
    data1 = data[data['model'] == model1]['value'].values  
    data2 = data[data['model'] == model2]['value'].values
    
    if len(data1) == 0 or len(data2) == 0:
        return None
    
    try:
        t_stat, p_val = stats.ttest_ind(data1, data2)
        cohens_d = _cohens_d(data1, data2)
        
        # Print result
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        label1, label2 = MODEL_LABELS.get(model1, model1), MODEL_LABELS.get(model2, model2)
        print(f"  {label1} vs {label2}: t={t_stat:.3f}, p={p_val:.4f}{sig}, d={cohens_d:.3f}")
        
        return {
            'criterion': criterion, 'model1': model1, 'model2': model2,
            'model1_label': label1, 'model2_label': label2,
            'model1_mean': np.mean(data1), 'model2_mean': np.mean(data2),
            'model1_std': np.std(data1, ddof=1), 'model2_std': np.std(data2, ddof=1),
            'model1_n': len(data1), 'model2_n': len(data2),
            'mean_diff': np.mean(data1) - np.mean(data2),
            't_statistic': t_stat, 'p_value': p_val, 'cohens_d': cohens_d,
            'significant_uncorrected': p_val < 0.05
        }
    except Exception:
        return None

def _cohens_d(data1, data2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(data1), len(data2)
    pooled_std = np.sqrt(((n1-1)*np.var(data1, ddof=1) + (n2-1)*np.var(data2, ddof=1)) / (n1+n2-2))
    return (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0

def _process_test_results(results, bonferroni_correction):
    """Process statistical results with optional correction."""
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    if bonferroni_correction:
        n_tests = len(df)
        df['p_value_corrected'] = (df['p_value'] * n_tests).clip(upper=1.0)
        df['significant_corrected'] = df['p_value_corrected'] < 0.05
        
        print(f"\n🔧 Bonferroni correction: {n_tests} tests, α = {0.05/n_tests:.6f}")
        print(f"   Significant: {sum(df['significant_uncorrected'])} → {sum(df['significant_corrected'])}")
    
    # Print summary
    p_col = 'p_value_corrected' if bonferroni_correction and 'p_value_corrected' in df.columns else 'p_value'
    sig_results = df[df[p_col] < 0.05]
    
    print(f"\n📊 Summary{' (Bonferroni corrected)' if bonferroni_correction else ''}:")
    if len(sig_results) > 0:
        for _, row in sig_results.iterrows():
            direction = ">" if row['mean_diff'] > 0 else "<"
            print(f"   {row['model1_label']} {direction} {row['model2_label']} on {row['criterion']}: "
                  f"p={row[p_col]:.4f}, d={row['cohens_d']:.3f}")
    else:
        print("   No significant differences (p < 0.05)")
    
    return df

# =============================================================================
# VISUALIZATION
# =============================================================================

def _create_plots(df_expanded, vis_dir, drop_incomplete):
    """Create all visualizations."""
    print("\n=== Creating Visualizations ===")
    _setup_plot_style()
    
    # Individual criterion plots
    regular_data = df_expanded[df_expanded['criterion'] != 'best-overall']
    if len(regular_data) > 0:
        for criterion in regular_data['criterion'].unique():
            _plot_criterion(regular_data, criterion, vis_dir, drop_incomplete)
    
    # Best-overall plot
    best_data = df_expanded[df_expanded['criterion'] == 'best-overall'] 
    if len(best_data) > 0:
        _plot_best_overall(best_data, vis_dir, drop_incomplete)
    
    # Combined overview
    _plot_combined(df_expanded, vis_dir, drop_incomplete)
    
    print("📊 All visualizations completed!")

def _setup_plot_style():
    """Configure matplotlib for publication-quality output."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Font selection
    serif_fonts = ['DejaVu Serif', 'Times', 'Times New Roman', 'Liberation Serif', 'serif']
    font = next((f for f in serif_fonts if f in plt.rcParams['font.serif'] or f == 'serif'), 'serif')
    
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif', 'font.serif': [font],
        'figure.dpi': 300, 'axes.linewidth': 0.8, 'grid.alpha': 0.3,
        'legend.frameon': True, 'legend.shadow': True, 'legend.framealpha': 0.9
    })

def _plot_criterion(regular_data, criterion, vis_dir, drop_incomplete):
    """Create individual criterion plot."""
    crit_data = regular_data[regular_data['criterion'] == criterion]
    models = [MODEL_LABELS[m] for m in MODEL_TYPES if MODEL_LABELS[m] in crit_data['model_label'].unique()]
    
    plt.figure(figsize=(10, 6))
    _create_boxplot(crit_data, 'model_label', 'value', models)
    _style_plot(f'{criterion}', 'Model', 'Score', crit_data)
    _save_plot(vis_dir, f"boxplot_{criterion.replace(' ', '_')}", drop_incomplete)

def _plot_best_overall(best_data, vis_dir, drop_incomplete):
    """Create best-overall plot.""" 
    plt.figure(figsize=(10, 6))
    _create_boxplot(best_data, 'model_label', 'value')
    _style_plot('Best Overall Model Ranking', 'Model', 'Rank (Lower = Better)', best_data, invert_y=True)
    _save_plot(vis_dir, "boxplot_best_overall", drop_incomplete)

def _plot_combined(df_expanded, vis_dir, drop_incomplete):
    """Create combined overview plot."""
    regular_data = df_expanded[df_expanded['criterion'] != 'best-overall']
    best_data = df_expanded[df_expanded['criterion'] == 'best-overall']
    
    criteria = regular_data['criterion'].unique() if len(regular_data) > 0 else []
    n_plots = len(criteria) + (1 if len(best_data) > 0 else 0)
    
    if n_plots == 0:
        return
    
    # Create layout
    if n_plots <= 2:
        rows, cols, figsize = 1, n_plots, (6 * n_plots, 6)
    elif n_plots <= 4:
        rows, cols, figsize = 2, 2, (12, 10)  
    else:
        rows, cols, figsize = (n_plots + 2) // 3, 3, (15, 5 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    
    plot_idx = 0
    
    # Plot criteria
    for criterion in criteria:
        _create_subplot(regular_data, criterion, axes[plot_idx])
        plot_idx += 1
    
    # Plot best-overall
    if len(best_data) > 0:
        _create_best_subplot(best_data, axes[plot_idx])
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Model Performance Comparison - All Criteria', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    _save_plot(vis_dir, "boxplot_combined", drop_incomplete)

def _create_boxplot(data, x, y, order=None):
    """Create styled boxplot with model-specific colors."""
    models = order or data[x].unique()
    colors = [MODEL_COLORS.get(model, MODEL_COLORS['default']) for model in models]
    
    box_plot = sns.boxplot(data=data, x=x, y=y, hue=x, order=order,
                          palette=colors, linewidth=1.2, fliersize=4, legend=False)
    
    # Style enhancements
    for patch in box_plot.artists:
        patch.set_alpha(0.8)
        patch.set_edgecolor('gray')
    
    for line in box_plot.lines[4::6]:  # Medians
        line.set_color('darkred')
        line.set_linewidth(2.5)
    
    for flier in box_plot.collections:  # Outliers
        flier.set_markerfacecolor('lightcoral')
        flier.set_alpha(0.7)
    
    return box_plot

def _style_plot(title, xlabel, ylabel, data, invert_y=False):
    """Apply plot styling."""
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')  
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    _add_sample_sizes(data)
    _add_means(data)
    
    if invert_y:
        plt.gca().invert_yaxis()
    
    plt.legend(loc='upper right', fontsize=11)
    plt.gca().set_facecolor('#FAFAFA')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

def _add_sample_sizes(data):
    """Add sample sizes to x-axis labels."""
    ax = plt.gca()
    labels = [label.get_text() for label in ax.get_xticklabels()]
    # new_labels = [f'{label}\n(n={len(data[data["model_label"] == label])})' for label in labels]
    new_labels = [f'{label}' for label in labels]
    ax.set_xticklabels(new_labels)

def _add_means(data):
    """Add mean markers."""
    means = data.groupby('model_label')['value'].mean()
    for i, model in enumerate(data['model_label'].unique()):
        if model in means.index:
            plt.plot(i, means[model], marker='D', color='darkred', markersize=10,
                    markeredgecolor='white', markeredgewidth=2, 
                    label='Mean' if i == 0 else "", zorder=10)

def _create_subplot(regular_data, criterion, ax):
    """Create subplot for criterion."""
    crit_data = regular_data[regular_data['criterion'] == criterion]
    models = [MODEL_LABELS[m] for m in MODEL_TYPES if MODEL_LABELS[m] in crit_data['model_label'].unique()]
    colors = [MODEL_COLORS.get(model, MODEL_COLORS['default']) for model in models]
    
    sns.boxplot(data=crit_data, x='model_label', y='value', hue='model_label',
               ax=ax, order=models, palette=colors, 
               linewidth=1.0, fliersize=3, legend=False)
    
    _style_subplot(ax, criterion, 'Score')
    _add_subplot_means(crit_data, models, ax)

def _create_best_subplot(best_data, ax):
    """Create subplot for best-overall."""
    models = best_data['model_label'].unique()
    colors = [MODEL_COLORS.get(model, MODEL_COLORS['default']) for model in models]
    
    sns.boxplot(data=best_data, x='model_label', y='value', hue='model_label',
               ax=ax, palette=colors, linewidth=1.0, fliersize=3, legend=False)
    
    _style_subplot(ax, 'Best Overall (Rank)', 'Rank (Lower = Better)')
    ax.invert_yaxis()
    _add_subplot_means(best_data, models, ax)

def _style_subplot(ax, title, ylabel):
    """Style subplot."""
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11) 
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.set_facecolor('#FAFAFA')
    ax.grid(True, alpha=0.3, linestyle='--')

def _add_subplot_means(data, models, ax):
    """Add mean markers to subplot."""
    means = data.groupby('model_label')['value'].mean()
    for i, model in enumerate(models):
        if model in means.index:
            ax.plot(i, means[model], marker='D', color='darkred', markersize=7,
                   markeredgecolor='white', markeredgewidth=1.5, zorder=10)

# =============================================================================
# UTILITIES
# =============================================================================

def _create_directories(*dirs):
    """Create directories.""" 
    for d in dirs:
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)

def _save_plot(vis_dir, filename, drop_incomplete):
    """Save plot as PDF."""
    suffix = '_complete_only' if drop_incomplete else '_all_rows'
    path = Path(vis_dir) / f"{filename}{suffix}.pdf"
    
    plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300,
                facecolor='white', pad_inches=0.1)
    print(f"✅ Plot saved: {path}")
    plt.close()

def _save_data(df_clean, df_expanded, ttest_results, output_dir, drop_incomplete):
    """Save all results."""
    suffix = '_complete_only' if drop_incomplete else '_all_rows'
    out_path = Path(output_dir)
    
    df_clean.to_csv(out_path / f'clean_data{suffix}.csv', index=False)
    df_expanded.to_csv(out_path / f'expanded_data{suffix}.csv', index=False) 
    
    if ttest_results is not None and len(ttest_results) > 0:
        ttest_results.to_csv(out_path / f'statistical_tests{suffix}.csv', index=False)
    
    print(f"💾 Data saved in: {out_path}")

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    df_expanded, df_clean, ttests = process_csv_complete_with_diagnostics_and_drop(
        'user_study_raw_results.csv',
        output_dir='results_user_study', 
        visualizations_dir='plots_user_study'
    )