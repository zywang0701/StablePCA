import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import umap


def compute_embedding(df, embedding_cols,
                      random_state=42,
                      perplexity=30,
                      n_neighbors=30, min_dist=0.3, metric='cosine'):
    """
    Compute both t-SNE and UMAP dimensionality reduction embeddings and return DataFrames with coordinates.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing embedding data and metadata, must include 'batch' and 'cell_type' columns
    embedding_cols : list of str
        Feature columns to use for dimensionality reduction (e.g., principal component columns)
    random_state : int, default 42
        Random seed for reproducibility
    perplexity : float, default 30
        t-SNE parameter, controls balance between local and global structure, typically between 5-50
    n_neighbors : int, default 30
        UMAP parameter, controls local neighborhood size
    min_dist : float, default 0.3
        UMAP parameter, controls minimum distance between points
    metric : str, default 'cosine'
        UMAP distance metric
    
    Returns:
    -------
    results : dict
        Dictionary with keys 'tsne' and 'umap', each containing:
        - 'df_plot': DataFrame with original data plus embedding coordinate columns
        - 'dim1_col': Name of the first dimension column
        - 'dim2_col': Name of the second dimension column
        - 'dim1_label': Label for the first dimension axis
        - 'dim2_label': Label for the second dimension axis
        - 'title_batch': Title for batch-colored plot
        - 'title_celltype': Title for cell type-colored plot
    """
    batch_col = 'batch'
    celltype_col = 'cell_type'
    
    # Check if required columns exist
    if batch_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{batch_col}' column")
    if celltype_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{celltype_col}' column")
    
    results = {}
    
    # Compute t-SNE dimensionality reduction
    tsne_result = TSNE(n_components=2, perplexity=perplexity, init='pca',
                      random_state=random_state).fit_transform(df[embedding_cols].values)
    df_plot_tsne = df.copy()
    df_plot_tsne['tSNE_1'] = tsne_result[:, 0]
    df_plot_tsne['tSNE_2'] = tsne_result[:, 1]
    
    results['tsne'] = {
        'df_plot': df_plot_tsne,
        'dim1_col': 'tSNE_1',
        'dim2_col': 'tSNE_2',
        'dim1_label': 't-SNE 1',
        'dim2_label': 't-SNE 2',
        'title_batch': 't-SNE by Batch',
        'title_celltype': 't-SNE by Cell Type'
    }
    
    # Compute UMAP dimensionality reduction
    umap_result = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    ).fit_transform(df[embedding_cols].values)
    df_plot_umap = df.copy()
    df_plot_umap['UMAP_1'] = umap_result[:, 0]
    df_plot_umap['UMAP_2'] = umap_result[:, 1]
    
    # Fill NaN cell types for UMAP
    df_plot_umap[celltype_col] = df_plot_umap[celltype_col].fillna("Unknown")
    
    results['umap'] = {
        'df_plot': df_plot_umap,
        'dim1_col': 'UMAP_1',
        'dim2_col': 'UMAP_2',
        'dim1_label': 'UMAP 1',
        'dim2_label': 'UMAP 2',
        'title_batch': 'UMAP by Batch',
        'title_celltype': 'UMAP by Cell Type'
    }
    
    return results


def plot_embedding(df_plot, dim1_col, dim2_col, dim1_label, dim2_label,
                   title_batch, title_celltype,
                   point_size=10, alpha=0.7,
                   palette_batch=None, palette_celltype=None,
                   show_plot=True):
    """
    Plot dimensionality reduction results with batch and cell type coloring.
    
    Parameters:
    ----------
    df_plot : pd.DataFrame
        DataFrame containing embedding coordinates and metadata ('batch' and 'cell_type' columns)
    dim1_col : str
        Name of the first dimension column in df_plot
    dim2_col : str
        Name of the second dimension column in df_plot
    dim1_label : str
        Label for the first dimension axis
    dim2_label : str
        Label for the second dimension axis
    title_batch : str
        Title for the batch-colored plot
    title_celltype : str
        Title for the cell type-colored plot
    point_size : int, default 10
        Size of points in the plot
    alpha : float, default 0.7
        Transparency of points, lower values are more transparent, typically between [0.3, 1.0]
    palette_batch : list or dict, default None
        Color palette for batch coloring
    palette_celltype : list or dict, default None
        Color palette for cell type coloring
    show_plot : bool, default True
        Whether to display the plot; if False, only return without displaying
    
    Returns:
    -------
    None
    """
    batch_col = 'batch'
    celltype_col = 'cell_type'
    
    if not show_plot:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 12), sharex=True, sharey=True)
    
    # Top plot: colored by batch
    sns.scatterplot(
        data=df_plot,
        x=dim1_col, y=dim2_col,
        hue=batch_col,
        s=point_size,
        alpha=alpha,
        palette=palette_batch,
        edgecolor=None,
        linewidth=0,
        ax=axes[0]
    )
    axes[0].set_title(title_batch, fontsize=14)
    axes[0].set_xlabel('')
    axes[0].set_ylabel(dim2_label)
    axes[0].legend(title=batch_col, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    # Bottom plot: colored by cell type
    sns.scatterplot(
        data=df_plot,
        x=dim1_col, y=dim2_col,
        hue=celltype_col,
        s=point_size,
        alpha=alpha,
        palette=palette_celltype,
        edgecolor=None,
        linewidth=0,
        ax=axes[1]
    )
    axes[1].set_title(title_celltype, fontsize=14)
    axes[1].set_xlabel(dim1_label)
    axes[1].set_ylabel(dim2_label)
    axes[1].legend(title=celltype_col, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_2x2_embedding(results, 
                       point_size=10, alpha=0.7,
                       palette_batch=None, palette_celltype=None,
                       figsize=(12, 10),
                       title_prefix="StablePCA k=50",
                       show_plot=True):
    """
    Plot 2x2 figure with t-SNE and UMAP, colored by batch (top row) and cell type (bottom row).
    Uses NeurIPS-style formatting similar to plot-singlecell.ipynb.
    Legends are shared: top row shares batch legend, bottom row shares cell_type legend.
    
    Parameters:
    ----------
    results : dict
        Dictionary from compute_embedding() with keys 'tsne' and 'umap'
    point_size : int, default 10
        Size of points in the plot
    alpha : float, default 0.7
        Transparency of points
    palette_batch : list or dict, default None
        Color palette for batch coloring
    palette_celltype : list or dict, default None
        Color palette for cell type coloring
    figsize : tuple, default (12, 10)
        Figure size (width, height)
    title_prefix : str, default "StablePCA k=50"
        Prefix for the figure title
    show_plot : bool, default True
        Whether to display the plot
    
    Returns:
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : numpy.ndarray
        Array of axes objects
    """
    # Apply NeurIPS-style formatting
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "figure.dpi": 600,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
    })
    
    batch_col = 'batch'
    celltype_col = 'cell_type'
    
    if not show_plot:
        return None, None
    
    # Extract data
    tsne_data = results['tsne']
    umap_data = results['umap']
    
    df_tsne = tsne_data['df_plot']
    df_umap = umap_data['df_plot']
    
    # Create 2x2 subplot with more space for titles and legends
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex='col', sharey='col')
    
    # Add column titles above top row (bold)
    # Position: left column center (0.25), right column center (0.75), near top (0.98)
    fig.text(0.25, 0.98, "t-SNE", ha='center', va='top', fontsize=18, weight='bold', 
             transform=fig.transFigure)
    fig.text(0.75, 0.98, "UMAP", ha='center', va='top', fontsize=18, weight='bold',
             transform=fig.transFigure)
    
    # Add row titles on the left (bold)
    # Position: top row center (0.75), bottom row center (0.25), left side (0.02)
    fig.text(0.02, 0.75, "by Batch", ha='left', va='center', fontsize=18, weight='bold',
             transform=fig.transFigure, rotation=90)
    fig.text(0.02, 0.25, "by Cell Type", ha='left', va='center', fontsize=18, weight='bold',
             transform=fig.transFigure, rotation=90)
    
    # Top row: batch coloring
    # Top left: t-SNE by batch
    ax1 = axes[0, 0]
    sns.scatterplot(
        data=df_tsne,
        x=tsne_data['dim1_col'], y=tsne_data['dim2_col'],
        hue=batch_col,
        s=point_size,
        alpha=alpha,
        palette=palette_batch,
        edgecolor=None,
        linewidth=0,
        ax=ax1,
        legend='auto',
        rasterized=True  # Rasterize scatter points for faster PDF rendering
    )
    ax1.set_title('')  # Remove title, using row label instead
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.grid(True, linestyle=":", linewidth=0.8, alpha=0.4)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    
    # Top right: UMAP by batch
    ax2 = axes[0, 1]
    sns.scatterplot(
        data=df_umap,
        x=umap_data['dim1_col'], y=umap_data['dim2_col'],
        hue=batch_col,
        s=point_size,
        alpha=alpha,
        palette=palette_batch,
        edgecolor=None,
        linewidth=0,
        ax=ax2,
        legend='auto',
        rasterized=True  # Rasterize scatter points for faster PDF rendering
    )
    ax2.set_title('')  # Remove title
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.grid(True, linestyle=":", linewidth=0.8, alpha=0.4)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    
    # Bottom row: cell type coloring
    # Bottom left: t-SNE by cell type
    ax3 = axes[1, 0]
    sns.scatterplot(
        data=df_tsne,
        x=tsne_data['dim1_col'], y=tsne_data['dim2_col'],
        hue=celltype_col,
        s=point_size,
        alpha=alpha,
        palette=palette_celltype,
        edgecolor=None,
        linewidth=0,
        ax=ax3,
        legend='auto',
        rasterized=True  # Rasterize scatter points for faster PDF rendering
    )
    ax3.set_title('')  # Remove title
    ax3.set_xlabel('')
    ax3.set_ylabel('')
    ax3.grid(True, linestyle=":", linewidth=0.8, alpha=0.4)
    for spine in ["top", "right"]:
        ax3.spines[spine].set_visible(False)
    
    # Bottom right: UMAP by cell type
    ax4 = axes[1, 1]
    sns.scatterplot(
        data=df_umap,
        x=umap_data['dim1_col'], y=umap_data['dim2_col'],
        hue=celltype_col,
        s=point_size,
        alpha=alpha,
        palette=palette_celltype,
        edgecolor=None,
        linewidth=0,
        ax=ax4,
        legend='auto',
        rasterized=True  # Rasterize scatter points for faster PDF rendering
    )
    ax4.set_title('')  # Remove title
    ax4.set_xlabel('')
    ax4.set_ylabel('')
    ax4.grid(True, linestyle=":", linewidth=0.8, alpha=0.4)
    for spine in ["top", "right"]:
        ax4.spines[spine].set_visible(False)
    
    # Share legends: remove individual legends and create shared ones
    # Get handles and labels from the plots
    handles_batch, labels_batch = ax1.get_legend_handles_labels()
    handles_celltype, labels_celltype = ax3.get_legend_handles_labels()
    
    # Remove individual legends
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.get_legend().remove()
    ax4.get_legend().remove()
    
    # Modify cell type labels: change 'monocyte' to 'Monocyte'
    labels_celltype_modified = [label.replace('monocyte', 'Monocyte') if label == 'monocyte' else label 
                                 for label in labels_celltype]
    
    # Add shared legend for top row (batch) - positioned on the right of first row
    # Position: right side (x=0.92), center of first row vertically (y=0.75)
    leg_batch = fig.legend(handles_batch, labels_batch, 
                          loc='center left', ncol=1, 
                          bbox_to_anchor=(0.92, 0.75), frameon=False, 
                          fontsize=13, title='Batch', title_fontsize=16,
                          handletextpad=0.3,  # Reduce space between marker and text
                          markerscale=2,    # Increase marker size
                          handlelength=1.8,   # Increase handle length for better alignment
                          columnspacing=0.5,  # Column spacing
                          alignment='left')   # Left align text
    
    # Add shared legend for bottom row (cell type) - positioned on the right of second row
    # Position: right side (x=0.92), center of second row vertically (y=0.25)
    leg_celltype = fig.legend(handles_celltype, labels_celltype_modified,
                             loc='center left', ncol=1,
                             bbox_to_anchor=(0.92, 0.25), frameon=False, 
                             fontsize=13, title='Cell Type', title_fontsize=16,
                             handletextpad=0.3,  # Reduce space between marker and text
                             markerscale=2,    # Increase marker size
                             handlelength=1.8,   # Increase handle length for better alignment
                             columnspacing=0.5,  # Column spacing
                             alignment='left')   # Left align text
    
    # Increase space between columns
    plt.tight_layout(rect=[0.05, 0, 0.92, 0.95])  # Leave space for left titles and right legends (reduced right margin)
    fig.subplots_adjust(wspace=0.2)  # Increase horizontal space between subplots
    plt.show()
    
    return fig, axes
