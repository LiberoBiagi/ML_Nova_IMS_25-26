
from scipy import stats
import numpy as np
from sklearn.feature_selection import VarianceThreshold,  RFE
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#FILTER METHOD


#ANOVA FUNCTION

def anova_for_categorical(df, y, categorical_cols):
    # Align indices between df and y
    common_idx = df.index.intersection(y.index)
    df_aligned = df.loc[common_idx]
    y_aligned = y.loc[common_idx]
    
    f_scores, p_values = [], []
    for col in df_aligned.columns:
        if col in categorical_cols:
            groups = [y_aligned[df_aligned[col] == cat] for cat in df_aligned[col].dropna().unique()]
            if len(groups) > 1 and all(len(g) > 1 for g in groups):
                f_stat, p_val = stats.f_oneway(*groups)
            else:
                f_stat, p_val = 0.0, 1.0
        else:
            if df_aligned[col].nunique() > 1:
                # Remove NaN values for correlation calculation
                valid_idx = df_aligned[col].notna() & y_aligned.notna()
                if valid_idx.sum() > 1:
                    corr = np.corrcoef(df_aligned.loc[valid_idx, col], y_aligned[valid_idx])[0, 1]
                    f_stat = corr**2 * valid_idx.sum()
                    p_val = 0.0
                else:
                    f_stat, p_val = 0.0, 1.0
            else:
                f_stat, p_val = 0.0, 1.0
        f_scores.append(f_stat)
        p_values.append(p_val)
    
    return np.array(f_scores), np.array(p_values)


def filter_method_selection(X_train, y_train, categorical_cols, num_cols,
                            top_k=None,
                            var_threshold=0.01,
                            corr_threshold=0.85):
    
    print("FILTER METHOD (Variance + Spearman Correlation + ANOVA)")
    
    # Ensure indices match
    common_idx = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common_idx]
    y_train = y_train.loc[common_idx]
    
    # Filter numerical columns that exist in X_train
    num_cols_in_X = [col for col in num_cols if col in X_train.columns]
    
    # Variance Threshold
    if num_cols_in_X:
        vt_selector = VarianceThreshold(threshold=var_threshold)
        X_num_vt = pd.DataFrame(
            vt_selector.fit_transform(X_train[num_cols_in_X]),
            columns=np.array(num_cols_in_X)[vt_selector.get_support()],
            index=X_train.index
        )
        print(f"Removed {len(num_cols_in_X) - X_num_vt.shape[1]} low-variance numeric features.")
    else:
        X_num_vt = pd.DataFrame(index=X_train.index)
    
    # Spearman Correlation
    if not X_num_vt.empty and X_num_vt.shape[1] > 1:
        corr_matrix = X_num_vt.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        X_num_corr = X_num_vt.drop(columns=to_drop)
        print(f"Removed {len(to_drop)} correlated numeric features (Spearman |corr| > {corr_threshold}).")
    else:
        X_num_corr = X_num_vt
    
    # Filter categorical columns that exist in X_train
    categorical_cols_in_X = [col for col in categorical_cols if col in X_train.columns]
    
    # Combine numeric + categorical
    X_filtered = pd.concat([X_num_corr, X_train[categorical_cols_in_X]], axis=1)
    
    # ANOVA F-test
    f_scores, f_pvalues = anova_for_categorical(X_filtered, y_train, categorical_cols_in_X)
    f_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-10)
    
    results_df = pd.DataFrame({
        'feature': X_filtered.columns,
        'ANOVA_F': f_scores,
        'ANOVA_p_value': f_pvalues,
        'ANOVA_norm': f_norm,
        'type': ['categorical' if c in categorical_cols_in_X else 'numerical' for c in X_filtered.columns]
    }).sort_values('ANOVA_norm', ascending=False)
    
    if top_k is None:
        selected_features = X_filtered.columns.tolist()
    else:
        selected_features = X_filtered.columns[np.argsort(f_norm)[-top_k:]].tolist()
    
    print(f"\nFilter method selected {len(selected_features)} features")
    
    return selected_features, X_filtered[selected_features], results_df

def rfe(X_train, X_val, y_train, y_val, num_cols, step=1, n_estimators=100, random_state=69):
    
    valid_num_cols = [col for col in num_cols if col in X_train.columns]
    if len(valid_num_cols) == 0:
        raise ValueError("No valid numeric columns found in X_train.")
    print(f"Using {len(valid_num_cols)} numeric columns for RFE:\n{valid_num_cols}")
    
    X_train_num = X_train[valid_num_cols]
    X_val_num = X_val[valid_num_cols]
    nof_list = np.arange(1, X_train_num.shape[1]+1)
    
    best_mae = float('inf')
    nof = 0
    train_mae_list = []
    val_mae_list = []
    features_to_select = None
    
    for n in nof_list:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        rfe = RFE(estimator=model, n_features_to_select=n, step=step)
        X_train_rfe = rfe.fit_transform(X_train_num, y_train)
        X_val_rfe = rfe.transform(X_val_num)
        
        model.fit(X_train_rfe, y_train)
        
        train_pred = model.predict(X_train_rfe)
        val_pred = model.predict(X_val_rfe)
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        train_mae_list.append(train_mae)
        val_mae_list.append(val_mae)
        
        if val_mae < best_mae:
            best_mae = val_mae
            nof = n
            features_to_select = pd.Series(rfe.support_, index=X_train_num.columns)
    
    selected_features = features_to_select[features_to_select].index.tolist()
    
    print(f"\nOptimum number of features: {nof}")
    print(f"Best validation MAE: {best_mae:.4f}")
    print("Selected features:")
    print(selected_features)
    
    return nof, best_mae, selected_features, train_mae_list, val_mae_list