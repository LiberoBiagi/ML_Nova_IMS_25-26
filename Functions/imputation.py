import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

def fit_mode_imputer(df):
    """
    Computes the most frequent Brand (mode) for each model in the dataframe.
    
    Args:
        df: Input DataFrame containing 'model_transformed' and 'Brand_transformed'.
        
    Returns:
        pd.Series: Mapping of model_transformed to the most frequent Brand_transformed.
    """
    # Get the most frequent brand for each model
    brand_models = df.groupby("model_transformed")["Brand_transformed"].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else pd.NA)
    return brand_models

def apply_mode_imputer(df, brand_models):
    """
    Impute missing Brand names using the pre-computed mode mapping.
    
    Args:
        df: Input DataFrame to impute.
        brand_models: Mapping of model_transformed to Brand_transformed (from fit_mode_imputer).
        
    Returns:
        pd.DataFrame: Dataframe with imputed 'Brand_transformed'.
    """
    df = df.copy() # Avoid modifying the original dataframe
    
    # Merge the brand modes into the dataframe
    df = pd.merge(df, brand_models, on="model_transformed", how="left", suffixes=("", "_mode"))
    
    # Fill NA values in Brand_transformed with the mode
    df["Brand_transformed"] = df["Brand_transformed"].fillna(df["Brand_transformed_mode"])
    
    # Remove the temporary column
    df.drop("Brand_transformed_mode", axis=1, inplace=True)
    
    return df

def fit_imputer(df): 
    """
    Fits an IterativeImputer using ExtraTreesRegressor on the dataframe.
    
    The 'price' column is temporarily removed during training if present, 
    since it shouldn't be used to impute features.
    
    Args:
        df: Training DataFrame.
        
    Returns:
        IterativeImputer: The fitted imputer object.
    """
    try: 
        df = df.drop(["price"], axis=1) # Remove the price column as it would not be available for imputation and thus should not be used when training the imputer
    except KeyError: # Happens when we use testing data as we previously dropped the price column 
        pass
    
    estimator = ExtraTreesRegressor(n_estimators=50,
                                    n_jobs=-1,
                                    min_samples_leaf=10,
                                    max_depth=20,
                                    random_state=69)

    imputer = IterativeImputer(estimator=estimator,
                                max_iter=20,
                                n_nearest_features=10,
                                random_state=69,
                                initial_strategy="most_frequent")

    imputer.fit(df)
    
    return imputer

def apply_imputer(df, imputer):
    """
    Apply the pretrained imputer to the dataframe features only, preserving the price column.
    
    This function handles separation of the price column, imputation, and reconstruction 
    of the DataFrame structure, as well as post-processing (rounding).
    
    Args:
        df: Input DataFrame to impute.
        imputer: Fitted IterativeImputer object.
        
    Returns:
        pd.DataFrame: Imputed DataFrame with original columns and index.
    """
    df_imputed = df.copy()
    
    # 1. Separate price if it exists
    price_col = None
    if 'price' in df_imputed.columns:
        price_col = df_imputed['price']
        df_for_imputation = df_imputed.drop(columns=['price'])
    else:
        df_for_imputation = df_imputed

    # 2. Apply imputer to features
    imputed_values = imputer.transform(df_for_imputation)
    
    # Create a new dataframe from the imputed values to avoid dtype warnings
    # We use the same index and columns as the input to preserve structure
    df_for_imputation = pd.DataFrame(
        imputed_values, 
        columns=df_for_imputation.columns, 
        index=df_for_imputation.index
    )
    
    # 3. Re-attach price if it existed
    if price_col is not None:
        df_for_imputation['price'] = price_col
        
    # 4. Post-processing (Rounding)
    df_for_imputation[["mpg", "engineSize"]] = abs(df_for_imputation[["mpg", "engineSize"]]).round(1)
    
    int_cols = ["year", "tax", "mileage", "previousOwners", "Brand_transformed", 
                "model_transformed", "transmission_transformed", "fuelType_transformed"]
    
    existing_int_cols = [col for col in int_cols if col in df_for_imputation.columns]
    df_for_imputation[existing_int_cols] = abs(df_for_imputation[existing_int_cols]).round(0).astype(int)
    
    return df_for_imputation

def fit_price_imputer(df, target_col='price'):
    """
    Trains a regressor to impute missing values in the target column (price).
    
    Args:
        df: Dataframe containing features and the target column with NaNs.
        target_col: The name of the target column to impute.
    
    Returns:
        model: The trained regression model (ExtraTreesRegressor).
    """
    
    # Separate data into sets with known and unknown target values
    train_known = df[df[target_col].notna()]
    
    # X contains all columns except the target
    X = train_known.drop(columns=[target_col])
    y = train_known[target_col]
    
    # Initialize the estimator (same robust parameters as your previous imputer)
    estimator = ExtraTreesRegressor(n_estimators=100,
                                    n_jobs=-1,
                                    min_samples_leaf=5,
                                    max_depth=20,
                                    random_state=69)
    
    # Fit the model
    estimator.fit(X, y)
    
    return estimator

def apply_price_imputer(df, model, target_col='price'):
    """
    Uses the trained model to fill NaN values in the target column.
    
    Args:
        df: DataFrame with missing target values.
        model: Trained regression model.
        target_col: Name of the target column.
        
    Returns:
        pd.DataFrame: DataFrame with target column imputed.
    """
    df_imputed = df.copy()
    
    # Identify rows where the target is missing
    missing_mask = df_imputed[target_col].isna()
    
    # If there are missing values, predict and fill them
    if missing_mask.sum() > 0:
        X_missing = df_imputed.loc[missing_mask].drop(columns=[target_col])
        predicted_values = model.predict(X_missing)
        df_imputed.loc[missing_mask, target_col] = predicted_values
        
    return df_imputed