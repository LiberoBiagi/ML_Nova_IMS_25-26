
def minimal_features(df):
    """
    Creates minimal engineering features based on domain knowledge.
    
    This function adds the following features:
    - age: 2020 - year
    - mileage_per_year: mileage / (age + 1)
    - efficiency_ratio: mpg / (engineSize + 0.1)
    - age_mileage: age * mileage / 100000
    - age_squared: age ** 2
    - is_premium_brand: Boolean integer indicating if the brand is BMW, Mercedes, or Audi.
    
    Args:
        df: Input DataFrame containing 'year', 'mileage', 'mpg', 'engineSize', and 'Brand'.
        
    Returns:
        pd.DataFrame: DataFrame with new features added and the original 'year' column dropped.
    """
    df = df.copy()
    df['age'] = 2020 - df['year']
    df = df.drop(columns=["year"])
    df['mileage_per_year'] = df['mileage'] / (df['age'] + 1)
    df['efficiency_ratio'] = df['mpg'] / (df['engineSize'] + 0.1)
    df['age_mileage'] = df['age'] * df['mileage'] / 100000
    df['age_squared'] = df['age'] ** 2
    df['is_premium_brand'] = df['Brand'].isin(['bmw', 'mercedes', 'audi']).astype(int)

    return df

def smoothed_mean_encoding(X_train, X_val, y_train, feature, m=10):
    """
    Applies smoothed mean encoding (Target Encoding) to a categorical feature.
    
    The smoothing formula used is:
    smoothed_mean = (count * mean + m * global_mean) / (count + m)
    
    This helps to handle rare categories by shrinking their encoding towards the global mean.
    
    Args:
        X_train: Training features DataFrame.
        X_val: Validation features DataFrame.
        y_train: Training target Series.
        feature: The name of the categorical column to encode.
        m: The smoothing parameter (weight of global mean). Default is 10.
        
    Returns:
        tuple: (X_train, X_val) with the new encoded column '{feature}_smoothed_mean_price'.
    """
    global_mean = y_train.mean()
    
    agg_stats = X_train.assign(target=y_train).groupby(feature)['target'].agg(['count', 'mean'])
    
    agg_stats['smoothed_mean'] = (
        (agg_stats['count'] * agg_stats['mean']) + (m * global_mean)
    ) / (agg_stats['count'] + m)
    

    mapping = agg_stats['smoothed_mean']
    

    X_train[f'{feature}_smoothed_mean_price'] = X_train[feature].map(mapping)
  
    X_val[f'{feature}_smoothed_mean_price'] = X_val[feature].map(mapping).fillna(global_mean)
    
    return X_train, X_val


def map_from_train_to_test(X_train, X_test, group_col, value_col, fill=-1):
    """
    Maps values from the training set to the test set based on a grouping column.
    
    Useful for applying aggregations (like mean price per model) calculated on train to test.
    
    Args:
        X_train: Training DataFrame.
        X_test: Test DataFrame.
        group_col: Column to group by (the key for mapping).
        value_col: Column to retrieve the value from (the value to map).
        fill: Value to fill if the group is missing in the test set. Default is -1.
        
    Returns:
        pd.DataFrame: Updates X_test in-place (and returns it) with mapped values in 'value_col'.
    """
    mapping = X_train.groupby(group_col)[value_col].first().to_dict()
    X_test[value_col] = X_test[group_col].map(mapping).fillna(fill)
    return X_test


