import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def univariate_analysis_cat(X_train, col):
    """Bar chart and pie chart for categorical features."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Count plot
    sns.countplot(data=X_train, x=col, palette='Set1', ax=ax[0])
    ax[0].set_title(f'Countplot for {col}')
    ax[0].tick_params(axis='x', rotation=45)

    # Pie chart
    data_counts = X_train[col].value_counts()
    ax[1].pie(
        data_counts,
        labels=data_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('pastel')
    )
    ax[1].set_title(f'Pie chart for {col}')

    plt.tight_layout()
    plt.show()

def bivariate_analysis_num(X_train, y_train, x_col, target_name='price'):
    """Scatter plot and box plot for numerical features against target."""
    fig = plt.figure(figsize=(14, 3))
    sns.set_palette("Set1")
    plt.suptitle(x_col, size=20, weight='bold')

    # Scatter plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_train[x_col], y=y_train)
    plt.xlabel(x_col)
    plt.ylabel(target_name)

    # Box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=X_train[x_col])
    plt.xlabel(x_col)

    plt.tight_layout()
    plt.show()

def spearman_corr_heatmap(X_train, y_train, target_name='price'):
    """Spearman correlation heatmap for numerical features including target."""
    # Select ONLY numeric columns
    df_corr = X_train.select_dtypes(include='number').copy()

    # Add target
    df_corr[target_name] = y_train

    corr = df_corr.corr(method='spearman')

    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.0)  # keep axis labels readable

    sns.heatmap(
        corr,
        annot=True,
        cmap='Set3',
        fmt=".2f",
        linewidths=1,
        linecolor='black',
        annot_kws={"size": 8} 
    )

    plt.title('Spearman Correlation Heatmap')

    plt.xticks(rotation=90) # make x-axis labels vertical
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()

def plot_category_mean_vs_global_avg(
    df,
    cat_col,
    cont_col,
    figsize=(14, 5),
    bar_color='steelblue',
    avg_line_color='coral'
):
    """
    Plot the mean of a continuous variable by category
    and compare it against the global average.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cat_col : str
        Categorical column name
    cont_col : str
        Continuous column name
    figsize : tuple, optional
        Figure size
    bar_color : str, optional
        Bar color
    avg_line_color : str, optional
        Global average line color
    """

    # Calculate category means
    category_means = (
        df.groupby(cat_col)[cont_col]
        .mean()
        .sort_values(ascending=False)
    )

    # Global average
    global_avg = df[cont_col].mean()

    # Create plot
    plt.figure(figsize=figsize)

    plt.bar(
        range(len(category_means)),
        category_means.values,
        color=bar_color,
        edgecolor='black',
        alpha=0.7,
        label='Category Average'
    )

    plt.axhline(
        y=global_avg,
        color=avg_line_color,
        linewidth=2.5,
        linestyle='--',
        label=f'Global Average: {global_avg:,.0f}'
    )

    # Labels & title
    plt.xlabel(cat_col, fontsize=12, fontweight='bold')
    plt.ylabel(f'Average {cont_col}', fontsize=12, fontweight='bold')
    plt.title(
        f'Average {cont_col} by {cat_col} vs Global Average',
        fontsize=14,
        fontweight='bold'
    )

    plt.xticks(
        range(len(category_means)),
        category_means.index,
        rotation=45,
        ha='right'
    )

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()