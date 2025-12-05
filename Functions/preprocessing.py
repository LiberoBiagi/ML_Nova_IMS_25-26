import pandas as pd
import numpy as np
from thefuzz import process
from sklearn.preprocessing import LabelEncoder

def simple_processing(df):
    """
    Apply string cleaning, brand/model corrections, and fuzzy matching to standardize the dataframe.
    
    This function performs the following operations:
    1. Converts string columns to lowercase and strips whitespace.
    2. Maps messy brand names to standard ones using a predefined mapping.
    3. Uses fuzzy matching to correct typos in 'model', 'transmission', and 'fuelType' columns.
    4. Rounds numeric columns like 'year', 'mileage', 'tax', etc.
    5. Creates a 'stated_no_damage' column from 'hasDamage' and drops the original.
    
    Args:
        df: Input pandas DataFrame containing car data.
        
    Returns:
        pd.DataFrame: The cleaned and standardized DataFrame.
    """
    df = df.copy()
    # ============================================================================
    # SECTION 1: REFERENCE DATA SETUP
    # ============================================================================
    
    # Reference list of correct model names
    models = ["golf", "veloste", "caddy", "yaris", "q2", "fiesta", "2 series", "3 series", "a3", "octavia", 
              "passat", "focus", "insignia", "a class", "q3", "fabia", "ka+", "glc class", "i30", "c class", 
              "polo", "e class", "q5", "up", "c-hr", "mokka x", "corsa", "astra", "tt", "5 series", "aygo", 
              "4 series", "slk", "viva", "t-roc", "ecosport", "tucson", "x-class", "cl class", "ix20", "i20", 
              "rapid", "a1", "auris", "sharan", "adam", "x3", "a8", "gls class", "b-max", "a4", "kona", "i10", 
              "mokka", "s-max", "x2", "crossland x", "tiguan", "a5", "gle class", "zafira", "ioniq", "a6", 
              "mondeo", "yeti outdoor", "x1", "scala", "s class", "1 series", "kamiq", "kuga", "tourneo connect", 
              "q7", "gla class", "arteon", "sl class", "santa fe", "grandland x", "i800", "rav4", "touran", 
              "citigo", "roomster", "prius", "corolla", "b class", "kodiaq", "v class", "caddy maxi life", 
              "superb", "getz", "combo life", "beetle", "galaxy", "m3", "gtc", "x4", "ka", "ix35", 
              "grand tourneo connect", "m4", "tourneo custom", "z4", "x5", "meriva", "rs6", "verso", "touareg", 
              "shuttle", "cls class", "c-max", "puma", "cla class", "i40", "tiguan allspace", "6 series", 
              "caravelle", "karoq", "i3", "grand c-max", "t-cross", "a7", "golf sv", "agila", "gt86", "yeti", 
              "california", "land cruiser", "edge", "x6", "caddy life", "8 series", "fusion", "gl class", 
              "scirocco", "z3", "proace verso", "hilux", "amarok", "cc", "7 series", "avensis", "eos", "m class", 
              "grandland", "zafira tourer", "rs5", "r8", "mustang", "antara", "q8", "camry", "clk", "rs3", 
              "jetta", "kadjar", "sq5", "rs4", "supra", "i8", "x7", "sq7", "g class", "s3", "crossland", 
              "tigra", "escort", "glb class", "vivaro", "verso-s", "m5", "s4", "iq", "a2", "caddy maxi", 
              "streetka", "cascada", "accent", "s8", "rs", "golf s", "ranger", "vectra", "ampera", "fox", 
              "urban cruiser", "m2", "clc class", "m6", "s5", "terracan", "200", "220", "230", "NaN"]
    
    # Get unique short model names (2 characters) for separate handling
    short_models = [models[i] for i in range(len(models)) if len(models[i]) == 2]
    short_models = list(set(short_models))
    
    transmission_types = ["semi-auto", "manual", "automatic", "unkown", "NaN", "other"]
    fuel_types = ["petrol", "diesel", "hybrid", "electric", "other", "NaN"]
    
    # Brand name corrections mapping
    brand_mapping = {
        "vw": "vw",
        "v": "vw",
        "w": "vw",
        
        "toyota": "toyota",
        "toyot": "toyota",
        "oyota": "toyota",
        
        "audi": "audi",
        "aud": "audi",
        "udi": "audi",
        "ud": "audi",
        
        "ford": "ford",
        "for": "ford",
        "ord": "ford",
        "or": "ford",
        
        "bmw": "bmw",
        "bm": "bmw",
        "mw": "bmw",
        
        "skoda": "skoda",
        "skod": "skoda",
        "koda": "skoda",
        "kod": "skoda",
        
        "opel": "opel",
        "ope": "opel",
        "pel": "opel",
        "pe": "opel",
        
        "mercedes": "mercedes",
        "mercede": "mercedes",
        "ercedes": "mercedes",
        "ercede": "mercedes",
        
        "hyundai": "hyundai",
        "hyunda": "hyundai",
        "yundai": "hyundai",
        "yunda": "hyundai"
    }
    
    # ============================================================================
    # SECTION 2: INITIAL CLEANING (NO FITTING REQUIRED) -> no risk of data leakage
    # ============================================================================
        
    # Convert brand names to lowercase and removes all beginning and trailing whitespace (e.g. space) from the column
    df["Brand"] = df["Brand"].str.lower().str.strip()
    df["model"] = df["model"].str.lower().str.strip()
    df["transmission"] = df["transmission"].str.lower().str.strip()
    df["fuelType"] = df["fuelType"].str.lower().str.strip()
    
    # Replace NaN with string NaN to avoid errors in the fuzzy algorithm (cant match NaNs)
    df[["model", "transmission", "fuelType"]] = df[["model", "transmission", "fuelType"]].fillna("NaN")
    
    # 1.1 Fixing brands
    df["Brand"] = df["Brand"].map(brand_mapping)
    
    # 1.2 Fixing models
    # Similarity matching for Models, Transmission and fuel columns (Fuzzy Match)
    # Source for process.extractOne (fuzzy): https://github.com/seatgeek/thefuzz
    
    # VECTORIZED APPROACH: Only perform fuzzy matching for unique value instead of per row
    
    # Models - handle different lengths separately
    unique_models = df["model"].unique()
    model_lookup = {}
    for val in unique_models:
        if pd.isna(val) or val == "NaN":
            model_lookup[val] = "NaN"
        elif len(val) > 2:  # Only perform fuzzy matching for models that have a name longer than 2 letters -> fuzzy will become fuzzy (unreliable) if names are to short
            model_lookup[val] = process.extractOne(val, models)[0]  # [0] because we get the name and score as a return -> score used for debugging
        elif len(val) == 2:  # Use the short names list for comparisons if the model names are 2 letters
            model_lookup[val] = process.extractOne(val, short_models)[0]
        else:  # We can define models with only one letter
            model_lookup[val] = "NaN"
    df["model"] = df["model"].map(model_lookup)
    
    # Transmission
    unique_trans = df["transmission"].unique()
    trans_lookup = {val: process.extractOne(val, transmission_types)[0] for val in unique_trans}
    df["transmission"] = df["transmission"].map(trans_lookup)
    
    # FuelType
    unique_fuel = df["fuelType"].unique()
    fuel_lookup = {val: process.extractOne(val, fuel_types)[0] for val in unique_fuel}
    df["fuelType"] = df["fuelType"].map(fuel_lookup)
    
    # Convert the str NaN values back to pd.NA for easier further processing and readability
    df["model"] = df["model"].replace("NaN", pd.NA)
    df["transmission"] = df["transmission"].replace(["unkown", "NaN", "other"], pd.NA)
    df["fuelType"] = df["fuelType"].replace(["other", "NaN"], pd.NA)
    
    ################################################################################
    # Simple Number Cleaning
    ################################################################################

    # Cleaning numeric columns
    df["year"] = df["year"].round(0)
    
    # Create the new column
    df["stated_no_damage"] = ~df["hasDamage"].astype(bool)
    df = df.drop(["hasDamage"], axis=1)

    # Round year to integer (no fractional years)
    df["year"] = df["year"].round()

    # Replace the years 2024, 2023 with NA as these values are very likley wrong (dataset is from 2020) and we want to impute them 
    df.loc[df["year"] == 2024, "year"] = pd.NA 
    df.loc[df["year"] == 2023, "year"] = pd.NA

    
    # Mileage: take absolute value and round
    # Some imputation might produce small negative values
    df["mileage"] = abs(df["mileage"].round())
    
    # Tax: take absolute value and round
    df["tax"] = abs(df["tax"].round())
    
    # MPG: round to 1 decimal place 
    df["mpg"] = abs(df["mpg"].round(1))
    
    # Engine size: round to 1 decimal place
    df["engineSize"] = abs(df["engineSize"].round(1))
    
    # Previous owners: take absolute value and round to integer
    df["previousOwners"] = abs(df["previousOwners"].round())
    
    return df

def remove_price_outliers(df, threshold=3):
    """
    Replaces price outliers with NaN for each model group based on the IQR method.
    
    This function groups data by 'model' and calculates the IQR for 'price'.
    Values outside (Q1 - threshold*IQR, Q3 + threshold*IQR) are set to NaN.
    
    Args:
        df: Input dataframe containing 'model' and 'price' columns.
        threshold: The multiplier for the IQR to define outliers. Default is 3 (extreme outliers).
    
    Returns:
        pd.DataFrame: Dataframe with outliers replaced by NaN.
    """
    df_clean = df.copy()
    
    # Calculate Q1, Q3, and IQR per model
    # We use transform to broadcast the group statistics back to the original index size
    groups = df_clean.groupby('model')['price']
    q1 = groups.transform(lambda x: x.quantile(0.25))
    q3 = groups.transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1
    
    # Define bounds
    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)
    
    # Identify outliers
    outlier_mask = (df_clean['price'] < lower_bound) | (df_clean['price'] > upper_bound)
    
    # Replace outliers with NaN
    df_clean.loc[outlier_mask, 'price'] = np.nan
    
    return df_clean

def fit_transform_encoding(df):
    """
    Fit label encoders for categorical columns on training data and transform the data.
    
    Handles missing values by only fitting on non-null values. Transformed columns
    are suffixed with '_transformed' and original columns are dropped.
    
    Args:
        df: Input pandas DataFrame.
        
    Returns:
        tuple: (pd.DataFrame, dict)
            - The encoded DataFrame.
            - A dictionary of fitted LabelEncoder objects.
    """
    
    encoders = {
        "Brand": LabelEncoder(),
        "model": LabelEncoder(),
        "transmission": LabelEncoder(),
        "fuelType": LabelEncoder()
    }

    columns = ["Brand", "model", "transmission", "fuelType"]

    # Code adapted from: https://stackoverflow.com/questions/36808434/label-encoder-encoding-missing-values

    for column in columns:
        # Get non-null string values
        mask = df[column].notna() & (df[column].apply(type) == str)
        
        # Fit encoder on unique non-null values
        fit_by = df.loc[mask, column].unique()
        encoders[column].fit(fit_by)
        
        # Transform only non-null values (vectorized)
        new_col_name = column + "_transformed"
        df[new_col_name] = pd.NA  # Initialize with NA
        df.loc[mask, new_col_name] = encoders[column].transform(df.loc[mask, column])
        
        # Convert to nullable integer
        df[new_col_name] = df[new_col_name].astype("Int64")

    df = df.drop(columns, axis=1)
    return df, encoders

def decode(df, encoders):
    """
    Decodes the integer coded columns back to their string representations.
    
    Also handles values that might be slightly outside the range (from imputation)
    by clipping them to the valid range of the encoder.
    
    Args:
        df: Input DataFrame with encoded columns.
        encoders: Dictionary of fitted LabelEncoder objects.
        
    Returns:
        pd.DataFrame: DataFrame with original string columns restored.
    """
    # Iterative imputer produces ~20-30 values that are outside of the range of the encoder
    # The simplest fix is to clip does values back into the range of the encoder
 

    df["Brand_transformed"] = df["Brand_transformed"].clip(lower=0, upper=encoders["Brand"].classes_.shape[0]-1).astype(int)
    df["Brand"] = encoders["Brand"].inverse_transform(df["Brand_transformed"])

    df["transmission_transformed"] = df["transmission_transformed"].clip(lower=0, upper=encoders["transmission"].classes_.shape[0]-1).astype(int)
    df["transmission"] = encoders["transmission"].inverse_transform(df["transmission_transformed"])
        
    # Use clip with the information of the fitted encoder, .classes_.shape gives us the dimension of the labels the encoder uses [0] is the rows - 1 because we start clipping at 0
    df["model_transformed"] = df["model_transformed"].clip(lower=0, upper=encoders["model"].classes_.shape[0]-1).astype(int)
    df["model"] = encoders["model"].inverse_transform(df["model_transformed"])
    
    df["fuelType_transformed"] = df["fuelType_transformed"].clip(lower=0, upper=encoders["fuelType"].classes_.shape[0]-1).astype(int)
    df["fuelType"] = encoders["fuelType"].inverse_transform(df["fuelType_transformed"])
    

    df.drop(columns=["Brand_transformed", "model_transformed", "transmission_transformed", "fuelType_transformed"], inplace=True)

    return df

def minimal_features(df):
    """
    Creates a minimal set of engineering features: 'age', 'mileage_per_year', 
    'efficiency_ratio', and 'age_mileage'.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with new features added and 'year' dropped.
    """
    df = df.copy()
    df['age'] = 2020 - df['year']
    df = df.drop(columns=["year"])
    df['mileage_per_year'] = df['mileage'] / (df['age'] + 1)
    df['efficiency_ratio'] = df['mpg'] / (df['engineSize'] + 0.1)
    df['age_mileage'] = df['age'] * df['mileage'] / 100000

    return df
