def minimal_features(df):
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
    mapping = X_train.groupby(group_col)[value_col].first().to_dict()
    X_test[value_col] = X_test[group_col].map(mapping).fillna(fill)
    return X_test


