# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Change 1: import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get('POLYTAB_ROOT', PACKAGE_DIR.parents[2]))
DATA_DIR = os.environ.get('POLYTAB_DATA_DIR', str(PROJECT_ROOT / 'data'))
RESULTS_DIR = os.environ.get('POLYTAB_OUTPUT_DIR', str(PROJECT_ROOT / 'results'))


def _basename_no_ext(path):
    base = os.path.basename(path)
    if '.' in base:
        base = '.'.join(base.split('.')[:-1])
    return base


# Change 2: class name changed
class SimpleETRPredictor:
    def __init__(self, n_estimators=500, n_jobs=-1, device=None, random_state=42, test_size=0.2):
        """
        ETR (Extra Trees Regressor) baseline model:
        1. Replace BayesianRidge with ExtraTreesRegressor
        2. Keep the pure tabular-feature input architecture
        3. n_estimators is the number of trees (hyperparameter)
        4. n_jobs controls parallel computation (-1 means using all cores)
        """
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        np.random.seed(random_state)

        # Keep standardization as requested
        self.scaler_Y_input = StandardScaler()
        self.scaler_Y_target = StandardScaler()

        self.models = {}  # Store ETR models, one per column
        self.Y_columns = []

        # Statistics storage
        self.test_size = test_size
        self.train_indices = None
        self.test_indices = None
        self._train_medians = None
        self.quantile_boundaries = None

        print(f"🔧 Core architecture: Scikit-learn ExtraTreesRegressor (ETR)")
        print(f"🌲 Parameter settings: n_estimators={self.n_estimators}, n_jobs={self.n_jobs}")
        print(f"🔀 Dataset split: train={1 - test_size:.0%}, test={test_size:.0%}")
        print(f"🔒 Strict mode: use only tabular features, without SMILES vectors")

    def _build_keep_mask(self, n_rows, n_cols, keep_k, rng):
        """Keep the original mask-generation logic unchanged"""
        keep_k = int(max(1, min(keep_k, n_cols - 1)))
        missing_mask_new = np.ones((n_rows, n_cols), dtype=bool)
        for i in range(n_rows):
            keep_cols = rng.choice(n_cols, size=keep_k, replace=False)
            missing_mask_new[i, keep_cols] = False
        return missing_mask_new, keep_k

    def _compute_quantile_boundaries(self, Y_data, missing_mask_new):
        """Keep the original quartile-computation logic unchanged"""
        quantile_boundaries = {}
        for j, col in enumerate(self.Y_columns):
            col_values = Y_data[:, j]
            observed_mask = ~missing_mask_new[:, j]
            observed_values = col_values[observed_mask]
            if len(observed_values) > 0:
                q_min = float(np.min(observed_values))
                q25 = float(np.percentile(observed_values, 25))
                q50 = float(np.percentile(observed_values, 50))
                q75 = float(np.percentile(observed_values, 75))
                q_max = float(np.max(observed_values))
            else:
                q_min = q25 = q50 = q75 = q_max = 0.0
            quantile_boundaries[col] = [q_min, q25, q50, q75, q_max]
        self.quantile_boundaries = quantile_boundaries
        return quantile_boundaries

    def _assign_class_labels(self, Y_data, missing_mask_new):
        """Keep interval-assignment logic"""
        n_rows, n_cols = Y_data.shape
        class_labels = np.zeros((n_rows, n_cols), dtype=np.int64)
        for j, col in enumerate(self.Y_columns):
            bounds = self.quantile_boundaries[col]
            col_values = Y_data[:, j]
            for i in range(n_rows):
                val = col_values[i]
                if val <= bounds[1]:
                    class_labels[i, j] = 0
                elif val <= bounds[2]:
                    class_labels[i, j] = 1
                elif val <= bounds[3]:
                    class_labels[i, j] = 2
                else:
                    class_labels[i, j] = 3
        return class_labels

    def load_and_process_data(self, csv_file_path, smiles_column='psmiles',
                              keep_mode='keep1', task_key=None):
        """
        Load data and perform a strict split.
        """
        print(f"📂 Load data: {csv_file_path}")
        df_raw = pd.read_csv(csv_file_path, encoding='utf-8-sig')

        # Identify valid numeric Y columns
        candidate_cols = [c for c in df_raw.columns if c != smiles_column]
        y_df_numeric = df_raw[candidate_cols].apply(pd.to_numeric, errors='coerce')
        df = df_raw.loc[~y_df_numeric.isnull().any(axis=1)].reset_index(drop=True)

        # Compute variance
        self.Y_columns = candidate_cols
        y_data = df[self.Y_columns].values.astype(np.float32)
        n_rows, n_cols = y_data.shape

        # 1. Split indices
        all_indices = np.arange(n_rows)
        self.train_indices, self.test_indices = train_test_split(
            all_indices, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )

        # 2. Masking strategy 
        if keep_mode == 'keep1':
            keep_k = 1
        elif keep_mode == 'keep2':  
            keep_k = 2
        elif keep_mode == 'keep3':  
            keep_k = 3    
        elif keep_mode == 'keep4':    
            keep_k = 4
        elif keep_mode == 'keep25':
            keep_k = int(np.rint(n_cols * 0.25))
        elif keep_mode == 'keep50':
            keep_k = int(np.rint(n_cols * 0.50))
        elif keep_mode == 'keep75':
            keep_k = int(np.rint(n_cols * 0.75))
        else:
            raise ValueError("keep_mode must be one of {'keep1','keep25','keep50','keep75'}")

        rng = np.random.RandomState(self.random_state)
        missing_mask_new, keep_k = self._build_keep_mask(n_rows, n_cols, keep_k, rng)

        # 3. Compute quantiles
        self._compute_quantile_boundaries(y_data[self.train_indices], missing_mask_new[self.train_indices])
        class_labels = self._assign_class_labels(y_data, missing_mask_new)

        # 4. Create output directory (change 3: automatically use the _ETR_ suffix)
        base_name = _basename_no_ext(csv_file_path)
        # Replace _Bayesian_ with _ETR_ here
        save_dir = os.path.join(RESULTS_DIR, f"{task_key if task_key else base_name}_ETR_{keep_mode}_260107_SAFE")
        os.makedirs(save_dir, exist_ok=True)

        # Save GT files
        df.iloc[self.train_indices].to_csv(os.path.join(save_dir, f"{base_name}_train_GT.csv"), index=False,
                                           encoding='utf-8-sig')
        df.iloc[self.test_indices].to_csv(os.path.join(save_dir, f"{base_name}_test_GT.csv"), index=False,
                                          encoding='utf-8-sig')

        return df, y_data, missing_mask_new, save_dir, class_labels

    def train_model(self, y_data, missing_mask_new):
        """
        Core training logic (ETR version, revised):
        1. Fill missing values using training-set medians (median imputation)
        2. Manually build StandardScaler and set var_ to avoid NotFittedError
        3. Train one ExtraTreesRegressor for each Y column
        """
        print(f"\n🚀 Starting Extra Trees model training (n_estimators={self.n_estimators}, n_jobs={self.n_jobs})...")
        print("ℹ️  Imputation strategy: training-set median")

        n_rows, n_cols = y_data.shape
        
        # --- 1. Compute medians and fill missing values ---
        y_data_filled = y_data.copy()
        self._train_medians = [] # Record medians

        # Process column by column
        for j in range(n_cols):
            # Get the training-set data and mask for this column
            col_train_values = y_data[self.train_indices, j]
            col_train_mask = missing_mask_new[self.train_indices, j]
            
            # Extract unmasked observed values
            observed_values = col_train_values[~col_train_mask]
            
            # Compute median
            if len(observed_values) > 0:
                med_val = float(np.median(observed_values))
            else:
                med_val = 0.0 # fallback strategy
            
            self._train_medians.append(med_val)

            # Fill all missing positions in this column, including train and test sets, with the computed median
            y_data_filled[missing_mask_new[:, j], j] = med_val

        # --- 2. Manually compute statistics and initialize scalers ---
        # Prepare data for mean/variance computation using visible positions only
        y_train_for_fit = y_data[self.train_indices].copy()
        train_mask = missing_mask_new[self.train_indices]
        y_train_for_fit[train_mask] = np.nan 

        means, scales = [], []
        for j in range(n_cols):
            # Extract non-NaN data for fitting
            obs = y_train_for_fit[:, j][~np.isnan(y_train_for_fit[:, j])].reshape(-1, 1)
            if len(obs) > 0:
                tmp = StandardScaler().fit(obs)
                means.append(tmp.mean_[0])
                scales.append(tmp.scale_[0])
            else:
                means.append(0.0)
                scales.append(1.0)
        
        # Convert to numpy arrays
        means_arr = np.array(means)
        scales_arr = np.array(scales)
        var_arr = scales_arr ** 2  # <--- Key fix: compute variance to avoid NotFittedError

        # --- Set input scaler attributes ---
        self.scaler_Y_input.mean_ = means_arr
        self.scaler_Y_input.scale_ = scales_arr
        self.scaler_Y_input.var_ = var_arr  # must be assigned
        self.scaler_Y_input.n_features_in_ = n_cols
        self.scaler_Y_input.n_samples_seen_ = np.array([len(self.train_indices)] * n_cols)

        # --- Set target scaler attributes ---
        self.scaler_Y_target.mean_ = means_arr
        self.scaler_Y_target.scale_ = scales_arr
        self.scaler_Y_target.var_ = var_arr # must be assigned
        self.scaler_Y_target.n_features_in_ = n_cols
        self.scaler_Y_target.n_samples_seen_ = np.array([len(self.train_indices)] * n_cols)

        # Apply transformation
        X_scaled = self.scaler_Y_input.transform(y_data_filled)
        Y_scaled_target = self.scaler_Y_target.transform(y_data)

        # --- 3. Train ExtraTreesRegressor models in a loop ---
        final_predictions_scaled = np.zeros_like(X_scaled)

        for j, col_name in enumerate(self.Y_columns):
            # Training target: ground-truth values of this column
            target_j = Y_scaled_target[self.train_indices, j]

            # Initialize ETR model
            model = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            )

            # Train the model: use imputed X to predict the target column
            model.fit(X_scaled[self.train_indices], target_j)

            self.models[col_name] = model
            # Predict the full dataset
            final_predictions_scaled[:, j] = model.predict(X_scaled)

            if (j + 1) % 5 == 0 or (j + 1) == n_cols:
                print(f"  [Progress] completed {j + 1}/{n_cols} column models trained")

        # Inverse-transform predictions to the original scale
        self.full_predictions = self.scaler_Y_target.inverse_transform(final_predictions_scaled)
        print("✅ ETR model training and full-set prediction complete")
        return self.full_predictions

    def perform_evaluation(self, y_true, y_pred, mask, indices, set_name="Test"):
        """General evaluation function: keep the original logic unchanged"""
        results = {}
        y_true_sub = y_true[indices]
        y_pred_sub = y_pred[indices]
        mask_sub = mask[indices]

        for j, col in enumerate(self.Y_columns):
            m = mask_sub[:, j]
            if m.sum() > 1:
                t_valid = y_true_sub[m, j]
                p_valid = y_pred_sub[m, j]

                valid_idx = ~(np.isnan(t_valid) | np.isnan(p_valid))
                if valid_idx.sum() > 1:
                    t, p = t_valid[valid_idx], p_valid[valid_idx]

                    mse = mean_squared_error(t, p)
                    mae = mean_absolute_error(t, p)
                    r2 = r2_score(t, p)
                    corr, p_val = pearsonr(t, p)
                    rel_err = np.mean(np.abs((p - t) / (np.abs(t) + 1e-8))) * 100

                    results[col] = {
                        'MSE': mse, 'MAE': mae, 'R²': r2,
                        'Correlation': corr, 'P-value': p_val,
                        'Relative_Error_%': rel_err,
                        f'N_{set_name.lower()}_masked': int(m.sum())
                    }

        print(f"\n🎯 {set_name} evaluation summary (masked region only):")
        if results:
            res_df = pd.DataFrame(results).T
            print(res_df[['R²', 'Correlation', 'MAE']].round(4))
            return res_df
        return pd.DataFrame()

    def visualize_train_results(self, train_predictions, y_data, missing_mask_new, save_dir="."):
        """Keep the original visualization logic unchanged"""
        print("📊 Generating training-set visualization at masked positions only...")
        train_Y_complete = y_data[self.train_indices]
        train_missing_mask = missing_mask_new[self.train_indices]
        train_pred_sub = train_predictions[self.train_indices]

        n_cols = 3
        n_rows = min(3, (len(self.Y_columns) + n_cols - 1) // n_cols)
        n_show = min(9, len(self.Y_columns))
        plt.figure(figsize=(15, 5 * n_rows))

        for i in range(n_show):
            col = self.Y_columns[i]
            plt.subplot(n_rows, n_cols, i + 1)
            mask = train_missing_mask[:, i]
            if mask.sum() > 0:
                t_valid = train_Y_complete[mask, i]
                p_valid = train_pred_sub[mask, i]
                plt.scatter(t_valid, p_valid, alpha=0.6, s=30, color='blue')

                limit_min = min(t_valid.min(), p_valid.min())
                limit_max = max(t_valid.max(), p_valid.max())
                plt.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', alpha=0.8)

                try:
                    r2 = r2_score(t_valid, p_valid)
                    corr, _ = pearsonr(t_valid, p_valid)
                except:
                    r2, corr = 0, 0

                plt.xlabel(f'True {col}')
                plt.ylabel(f'Predicted {col}')
                plt.title(f'{col} (Train - Masked)\nR² = {r2:.3f}, Corr = {corr:.3f}')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'train_set_predictions_masked.png'), dpi=300)
        plt.show()

    def visualize_test_results(self, test_predictions, y_data, missing_mask_new, save_dir="."):
        """Keep the original visualization logic unchanged"""
        print("📊 Generating test-set visualization...")
        test_Y_complete = y_data[self.test_indices]
        test_missing_mask = missing_mask_new[self.test_indices]
        test_pred_sub = test_predictions[self.test_indices]

        n_cols = 3
        n_rows = min(3, (len(self.Y_columns) + n_cols - 1) // n_cols)
        n_show = min(9, len(self.Y_columns))
        plt.figure(figsize=(15, 5 * n_rows))

        for i in range(n_show):
            col = self.Y_columns[i]
            plt.subplot(n_rows, n_cols, i + 1)
            mask = test_missing_mask[:, i]
            if mask.sum() > 0:
                t_valid = test_Y_complete[mask, i]
                p_valid = test_pred_sub[mask, i]
                plt.scatter(t_valid, p_valid, alpha=0.6, s=30, color='red')

                limit_min = min(t_valid.min(), p_valid.min())
                limit_max = max(t_valid.max(), p_valid.max())
                plt.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', alpha=0.8)

                try:
                    r2 = r2_score(t_valid, p_valid)
                    corr, _ = pearsonr(t_valid, p_valid)
                except:
                    r2, corr = 0, 0

                plt.xlabel(f'True {col}')
                plt.ylabel(f'Predicted {col}')
                plt.title(f'{col} (Test Set)\nR² = {r2:.3f}, Corr = {corr:.3f}')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'test_set_predictions.png'), dpi=300)
        plt.show()

    def plot_training_curve(self, save_dir="."):
        """Updated text description to indicate that ETR is used"""
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Extra Trees Regressor: No iterative training curve",
                 ha='center', va='center', fontsize=12)
        plt.savefig(os.path.join(save_dir, 'training_curve.png'), dpi=300)
        plt.close()

    def run_pipeline(self, csv_file_path, smiles_column='psmiles', keep_mode='keep1', task_key=None):
        """Full pipeline runner"""
        print("=" * 60)
        print(f"🧪 ETR Baseline prediction system | strategy: {keep_mode}")
        print("=" * 60)

        # 1. Load data
        df, y_data, missing_mask_new, save_dir, class_labels = self.load_and_process_data(
            csv_file_path, smiles_column, keep_mode, task_key
        )
        base_name = _basename_no_ext(csv_file_path)

        # 2. Train and obtain full-set predictions
        predictions = self.train_model(y_data, missing_mask_new)

        # 3. Evaluate
        train_res = self.perform_evaluation(y_data, predictions, missing_mask_new, self.train_indices, "Train")
        test_res = self.perform_evaluation(y_data, predictions, missing_mask_new, self.test_indices, "Test")

        # 4. Visualize and save
        self.visualize_train_results(predictions, y_data, missing_mask_new, save_dir)
        self.visualize_test_results(predictions, y_data, missing_mask_new, save_dir)
        self.plot_training_curve(save_dir)

        # Save CSV results
        if not train_res.empty:
            train_res.to_csv(os.path.join(save_dir, 'train_set_evaluation_masked.csv'), encoding='utf-8-sig')
        if not test_res.empty:
            test_res.to_csv(os.path.join(save_dir, 'test_set_evaluation.csv'), encoding='utf-8-sig')

        # 5. Performance comparison summary
        if not train_res.empty and not test_res.empty:
            comparison = pd.DataFrame({
                'Train_R²': train_res['R²'], 'Test_R²': test_res['R²'],
                'Train_Corr': train_res['Correlation'], 'Test_Corr': test_res['Correlation'],
                'Train_MAE': train_res['MAE'], 'Test_MAE': test_res['MAE']
            })
            comparison.to_csv(os.path.join(save_dir, 'train_test_comparison.csv'), encoding='utf-8-sig')
            print("\n📊 Performance comparison (R²):")
            print(comparison[['Train_R²', 'Test_R²']].round(4))

        # 6. Save detailed prediction results
        # Test-set predictions
        test_df = df.iloc[self.test_indices].copy()
        test_mask = missing_mask_new[self.test_indices]
        for j, col in enumerate(self.Y_columns):
            vals = test_df[col].values.copy()
            vals[test_mask[:, j]] = predictions[self.test_indices, j][test_mask[:, j]]
            test_df[col] = vals
        test_df.to_csv(os.path.join(save_dir, f"{base_name}_test_predictions.csv"), index=False, encoding='utf-8-sig')

        # Training-set predictions
        train_df = df.iloc[self.train_indices].copy()
        train_mask = missing_mask_new[self.train_indices]
        for j, col in enumerate(self.Y_columns):
            vals = train_df[col].values.copy()
            vals[train_mask[:, j]] = predictions[self.train_indices, j][train_mask[:, j]]
            train_df[col] = vals
        train_df.to_csv(os.path.join(save_dir, f"{base_name}_train_predictions.csv"), index=False, encoding='utf-8-sig')

        print(f"\n✅ Pipeline complete! Output directory: {save_dir}")


# --- Example run ---
if __name__ == "__main__":
    # Initialize the ETR predictor
    # n_estimators=100 (default number of trees), n_jobs=-1 (parallel computation), random_state=42 (fixed seed)
    predictor = SimpleETRPredictor(n_estimators=100, n_jobs=-1, random_state=42, test_size=0.2)

    # Task configuration
    data_dict = {'md': 'MD_properties_simple.csv'}
    smiles_col_dict = {'md': 'SMILES'}

    task_key = 'md'
    csv_path = os.path.join(DATA_DIR, data_dict[task_key])
    smiles_col = smiles_col_dict[task_key]

    # Test different retention strategies; the original structure is kept, and 'keep4' can be added directly if needed
    for km in ['keep1', 'keep2','keep3', 'keep4', 'keep25', 'keep50', 'keep75']:
        predictor.run_pipeline(csv_file_path=csv_path, smiles_column=smiles_col, keep_mode=km, task_key=task_key)


