import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import random
import time

class BrandModelTrainer:
    """
    A unified interface for training and evaluating models separately for each car brand.
    
    This class handles the scaling (RobustScaler) and encoding (OneHotEncoder) pipeline 
    automatically for each brand's subset of data.
    
    Attributes:
        estimator: The base sklearn estimator to be cloned for each brand.
        brand_models: Dictionary storing the trained pipeline for each brand.
        feature_cols: List of feature names used for training.
    """
    def __init__(self, estimator):
        self.estimator = estimator
        self.brand_models = {}
        self.feature_cols = None

    def fit(self, X_train, y_train):
        """
        Fits a separate model pipeline for each unique brand in X_train.
        
        Args:
            X_train: Training features DataFrame containing a 'Brand' column.
            y_train: Training target Series.
            
        Returns:
            self: The fitted BrandModelTrainer instance.
        """
        self.feature_cols = [c for c in X_train.columns if c != "Brand"]
        print(f"Training models for {len(X_train['Brand'].unique())} brands...\n")

        for brand in X_train["Brand"].unique():
            mask = X_train["Brand"] == brand
            Xb = X_train.loc[mask, self.feature_cols]
            yb = y_train[mask]

            numeric_cols = Xb.select_dtypes(include=["int64", "int32" ,"float64"]).columns.tolist()
            categorical_cols = Xb.select_dtypes(include=["object", "category"]).columns.tolist()

            preprocessor = ColumnTransformer([
                ("num", RobustScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
            ])

            model = Pipeline([
                ("preprocess", preprocessor),
                ("estimator", clone(self.estimator))
            ])

            model.fit(Xb, yb)
            self.brand_models[brand] = model
            print(f"  ✓ {brand} done.")

        return self

    def predict(self, X):
        """
        Predicts target values for the input data using the brand-specific models.
        
        Args:
            X: Input DataFrame containing a 'Brand' column.
            
        Returns:
            np.array: Array of predicted values. Zero for brands not seen during training.
        """
        preds = np.zeros(len(X))
        for brand, model in self.brand_models.items():
            mask = X["Brand"] == brand
            if mask.sum() == 0:
                continue
            Xb = X.loc[mask, self.feature_cols]
            preds[mask] = model.predict(Xb)
        return preds

    def evaluate_train(self, X_train, y_train):
        """
        Evaluates the model on the training set and prints performance metrics.
        
        Args:
            X_train: Training features DataFrame.
            y_train: Training target Series.
            
        Returns:
            dict: Dictionary containing RMSE, MAE, and R².
        """
        y_pred = self.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        print("\nTraining Set Performance (Overall):")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE:  {mae:.2f}")
        print(f"  R²:   {r2:.4f}")
        return {"RMSE": rmse, "MAE": mae, "R²": r2}

    def evaluate(self, X_val, y_val):
        """
        Evaluates the model on the validation set and prints performance metrics.
        
        Args:
            X_val: Validation features DataFrame.
            y_val: Validation target Series.
            
        Returns:
            dict: Dictionary containing RMSE, MAE, and R².
        """
        y_pred = self.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        print("\nValidation Set Performance (Overall):")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE:  {mae:.2f}")
        print(f"  R²:   {r2:.4f}")
        return {"RMSE": rmse, "MAE": mae, "R²": r2}

    def evaluate_by_brand(self, X, y, split_name="Validation"):
        """
        Calculates and prints performance metrics broken down by brand.
        
        Args:
            X: Input DataFrame.
            y: Target Series.
            split_name: Name of the data split (for display). Default is "Validation".
            
        Returns:
            pd.DataFrame: DataFrame containing metrics per brand.
        """
        y_pred = self.predict(X)
        results = []
        for brand in X["Brand"].unique():
            mask = X["Brand"] == brand
            y_true_b = y[mask]
            y_pred_b = y_pred[mask]
            rmse = np.sqrt(mean_squared_error(y_true_b, y_pred_b))
            mae = mean_absolute_error(y_true_b, y_pred_b)
            r2 = r2_score(y_true_b, y_pred_b)
            results.append({"Brand": brand, "N": len(y_true_b), "RMSE": rmse, "MAE": mae, "R²": r2})
        df = pd.DataFrame(results).sort_values("RMSE")
        print(f"\n{split_name} Performance per Brand:")
        print(df.to_string(index=False))
        return df

    def evaluate_train_by_brand(self, X_train, y_train):
        """
        Calculates and prints training performance metrics per brand.
        
        Args:
            X_train: Training features DataFrame.
            y_train: Training target Series.
            
        Returns:
            pd.DataFrame: DataFrame containing metrics per brand.
        """
        return self.evaluate_by_brand(X_train, y_train, split_name="Training")
    
    def save_predictions(self, X, path):
        """
        Generates predictions and saves them to a CSV file.
        
        Args:
            X: Input DataFrame containing 'CarID' and features.
            path: Path to save the CSV file.
            
        Returns:
            pd.DataFrame: The DataFrame that was saved.
        """
        preds = self.predict(X)
        df = pd.DataFrame({
            "CarID": X["CarID"].values,
            "price": preds
        })
        df.to_csv(path, index=False)
        print(f"File salvato in: {path}")
        return df


class HoldoutRandomSearch:
    """
    Performs randomized hyperparameter search using a fixed holdout validation set.
    
    Optimizes model hyperparameters specifically for the BrandModelTrainer wrapper.
    Tracks the best configuration globally and per-brand.
    
    Attributes:
        trainer_class: The class to instantiate (e.g., BrandModelTrainer).
        param_space: Dictionary or list of hyperparameter distributions.
        n_iter: Number of random configurations to test.
        optimize_metric: Metric to minimize ('mae' or 'rmse').
    """
    def __init__(self, trainer_class, param_space, n_iter=20, optimize_metric='mae'):
        self.trainer_class = trainer_class
        self.param_space = param_space
        self.n_iter = n_iter
        self.optimize_metric = optimize_metric.lower()
        self.results = []
        self.best_params = None
        self.best_score = None
        self.best_trainer = None
        self.brand_best_configs = {}
        
    def sample_params(self):
        """
        Randomly samples a parameter configuration from the parameter space.
        
        Returns:
            dict: Sampled hyperparameters.
        """
        if isinstance(self.param_space, list):
            return random.choice(self.param_space)
        else:
            return {k: random.choice(v) for k, v in self.param_space.items()}
    
    def _get_param_summary(self, params):
        summary = {}
        
        if 'n_estimators' in params:
            summary['n_estimators'] = params['n_estimators']
        if 'max_depth' in params:
            summary['max_depth'] = params['max_depth']
        if 'max_features' in params:
            summary['max_features'] = params['max_features']
            
        if 'hidden_layer_sizes' in params:
            summary['hidden_layers'] = str(params['hidden_layer_sizes'])
        if 'alpha' in params:
            summary['alpha'] = params['alpha']
        if 'learning_rate_init' in params:
            summary['lr'] = params['learning_rate_init']
        if 'activation' in params:
            summary['activation'] = params['activation']
            
        return summary
    
    def run(self, X_train, y_train, X_val, y_val):
        """
        Executes the random search process.
        
        Iterates n_iter times, training a new model with sampled parameters, evaluating on
        the validation set, and updating best scores.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            
        Returns:
            tuple: (best_trainer_instance, best_params_dict, best_score)
        """
        metric_name = self.optimize_metric.upper()
        print(f"Running random search ({self.n_iter} iterations)...")
        print(f"Optimizing for: {metric_name}\n")
        
        start_time = time.time()
        brands = X_train["Brand"].unique()

        for brand in brands:
            self.brand_best_configs[brand] = {
                'score': float('inf'),
                'params': None,
                'config_num': None,
                'all_metrics': {}
            }
        
        for i in range(self.n_iter):
            iter_start = time.time()
            params = self.sample_params()
            
            print(f"\n{'='*70}")
            print(f"[{i+1}/{self.n_iter}] Testing params:")
            print(params)
            print('='*70)
            

            estimator = self.trainer_class.estimator.__class__(**params)
            trainer = BrandModelTrainer(estimator)
            trainer.fit(X_train, y_train)

            metrics = trainer.evaluate(X_val, y_val)
            rmse = metrics["RMSE"]
            mae = metrics["MAE"]
            r2 = metrics["R²"]
            
            current_score = mae if self.optimize_metric == 'mae' else rmse
            
            brand_results = {}
            for brand in brands:
                mask = X_val["Brand"] == brand
                y_true_brand = y_val[mask]
                y_pred_brand = trainer.predict(X_val[mask])
                
                brand_rmse = np.sqrt(mean_squared_error(y_true_brand, y_pred_brand))
                brand_mae = mean_absolute_error(y_true_brand, y_pred_brand)
                brand_r2 = r2_score(y_true_brand, y_pred_brand)
                
                brand_score = brand_mae if self.optimize_metric == 'mae' else brand_rmse
                
                brand_results[brand] = {
                    'score': brand_score,
                    'rmse': brand_rmse,
                    'mae': brand_mae,
                    'r2': brand_r2
                }
                
                if brand_score < self.brand_best_configs[brand]['score']:
                    self.brand_best_configs[brand]['score'] = brand_score
                    self.brand_best_configs[brand]['params'] = params.copy()
                    self.brand_best_configs[brand]['config_num'] = i + 1
                    self.brand_best_configs[brand]['all_metrics'] = {
                        'rmse': brand_rmse,
                        'mae': brand_mae,
                        'r2': brand_r2
                    }
            
            self.results.append({
                "config_num": i + 1,
                "params": params,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "score": current_score,
                "brand_results": brand_results
            })

            if self.best_score is None or current_score < self.best_score:
                self.best_score = current_score
                self.best_params = params
                self.best_trainer = trainer
                print(f"New best overall model ({metric_name}: {current_score:.2f})")
           

            elapsed_total = time.time() - start_time
            avg_per_iter = elapsed_total / (i + 1)
            eta = avg_per_iter * (self.n_iter - (i + 1))
            
            print(f"\nProgress: {i+1}/{self.n_iter} | Elapsed: {elapsed_total:.1f}s | ETA: ~{eta:.1f}s")
        
        print(f"SEARCH COMPLETED - FINAL RESULTS (Optimized for {metric_name})")
        
        print(f"\n Best general model:")
        print(f"  Best {metric_name}: {self.best_score:.2f}")
        
        best_result = [r for r in self.results if r['score'] == self.best_score][0]
        print(f"  RMSE: {best_result['rmse']:.2f}")
        print(f"  MAE:  {best_result['mae']:.2f}")
        print(f"  R²:   {best_result['r2']:.4f}")
        print(f"  Params: {self.best_params}")
   
        print(f"Best configuration per brand (by {metric_name})")

        
        brand_summary = []
        for brand in sorted(brands):
            config = self.brand_best_configs[brand]
            
            summary = {
                'Brand': brand,
                f'Best_{metric_name}': config['score'],
                'RMSE': config['all_metrics']['rmse'],
                'MAE': config['all_metrics']['mae'],
                'R²': config['all_metrics']['r2'],
                'Config_Num': config['config_num']
            }
            
            param_summary = self._get_param_summary(config['params'])
            summary.update(param_summary)
            
            brand_summary.append(summary)
            
            print(f"\n{brand.upper()}:")
            print(f"  Best {metric_name}: {config['score']:.2f}")
            print(f"  RMSE: {config['all_metrics']['rmse']:.2f}")
            print(f"  MAE:  {config['all_metrics']['mae']:.2f}")
            print(f"  R²:   {config['all_metrics']['r2']:.4f}")
            print(f"  Found at iteration: {config['config_num']}")
            print(f"  Best params:")
            for k, v in list(config['params'].items())[:5]:
                if k not in ['random_state', 'n_jobs', 'shuffle', 'verbose', 'warm_start']:
                    print(f"    {k}: {v}")
        
        df_summary = pd.DataFrame(brand_summary).sort_values(f'Best_{metric_name}')
  
        print(df_summary.to_string(index=False))
        
        results_df = pd.DataFrame([
            {
                'Config': r['config_num'],
                metric_name: r['score'],
                'RMSE': r['rmse'],
                'MAE': r['mae'],
                'R²': r['r2']
            }
            for r in self.results
        ]).sort_values(metric_name)
        
        print(f"ALL CONFIGURATIONS (sorted by {metric_name}):")
        print(results_df.head(10).to_string(index=False))
        
        return self.best_trainer, self.best_params, self.best_score