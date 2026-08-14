# -*- coding: utf-8 -*-
# Auto-generated from BLR.ipynb

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings

# 移除 torch 相关导入，改为纯 sklearn 架构
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

class SimpleBayesianYPredictor:
    def __init__(self, device=None, random_state=42, test_size=0.2):
        """
        贝叶斯基准模型：
        1. 移除 PolyBERT 特征提取
        2. Remove the main-model attention and cascade-regression modules
        3. 仅使用表格 Y 值的 Mask 状态作为特征
        """
        self.random_state = random_state
        np.random.seed(random_state)

        self.scaler_Y_input = StandardScaler()
        self.scaler_Y_target = StandardScaler()

        self.models = {}  # 存储 BayesianRidge 模型 (每列一个)
        self.Y_columns = []
        self.low_variance_cols = []

        # 统计量存储（严格遵循训练集）
        self.test_size = test_size
        self.train_indices = None
        self.test_indices = None
        self._train_medians = None
        self.quantile_boundaries = None

        print(f"🔧 核心架构: Scikit-learn BayesianRidge (Baseline)")
        print(f"🔀 数据集划分: 训练集={1 - test_size:.0%}, 测试集={test_size:.0%}")
        print(f"🔒 严格模式：仅使用表格特征，无 SMILES 向量介入")

    def _build_keep_mask(self, n_rows, n_cols, keep_k, rng):
        """完全保留 126.py 的掩码生成逻辑"""
        keep_k = int(max(1, min(keep_k, n_cols - 1)))
        missing_mask_new = np.ones((n_rows, n_cols), dtype=bool)
        for i in range(n_rows):
            keep_cols = rng.choice(n_cols, size=keep_k, replace=False)
            missing_mask_new[i, keep_cols] = False
        return missing_mask_new, keep_k

    def _compute_quantile_boundaries(self, Y_data, missing_mask_new):
        """完全保留 126.py 的四分位数计算逻辑，用于后续分类文件保存"""
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
        """保留区间分配逻辑，虽然不用分类器，但用于生成对比用的 class_probabilities 文件"""
        n_rows, n_cols = Y_data.shape
        class_labels = np.zeros((n_rows, n_cols), dtype=np.int64)
        for j, col in enumerate(self.Y_columns):
            bounds = self.quantile_boundaries[col]
            col_values = Y_data[:, j]
            for i in range(n_rows):
                val = col_values[i]
                if val <= bounds[1]: class_labels[i, j] = 0
                elif val <= bounds[2]: class_labels[i, j] = 1
                elif val <= bounds[3]: class_labels[i, j] = 2
                else: class_labels[i, j] = 3
        return class_labels

    def load_and_process_data(self, csv_file_path, smiles_column='psmiles',
                              keep_mode='keep1', task_key=None):
        """
        加载数据并进行严格划分。
        移除了 BERT 相关的 features 提取步骤。
        """
        print(f"📂 加载数据: {csv_file_path}")
        df_raw = pd.read_csv(csv_file_path, encoding='utf-8-sig')

        # 识别有效的数值 Y 列
        candidate_cols = [c for c in df_raw.columns if c != smiles_column]
        y_df_numeric = df_raw[candidate_cols].apply(pd.to_numeric, errors='coerce')
        df = df_raw.loc[~y_df_numeric.isnull().any(axis=1)].reset_index(drop=True)

        # 统计方差（保留 126.py 的过滤逻辑）
        self.Y_columns = candidate_cols
        y_data = df[self.Y_columns].values.astype(np.float32)
        n_rows, n_cols = y_data.shape

        # 1. 划分索引 (严格遵循原代码顺序)
        all_indices = np.arange(n_rows)
        self.train_indices, self.test_indices = train_test_split(
            all_indices, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )

        # 2. 掩码策略 (完全对齐原代码)
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
            raise ValueError("keep_mode 必须是 {'keep1','keep25','keep50','keep75'}")
        rng = np.random.RandomState(self.random_state)
        missing_mask_new, keep_k = self._build_keep_mask(n_rows, n_cols, keep_k, rng)

        # 3. 计算分位数 (仅基于训练集保留位)
        self._compute_quantile_boundaries(y_data[self.train_indices], missing_mask_new[self.train_indices])
        class_labels = self._assign_class_labels(y_data, missing_mask_new)

        # 4. 目录创建 (为了区分原实验，增加 _Bayesian 后缀)
        base_name = _basename_no_ext(csv_file_path)
        save_dir = os.path.join(RESULTS_DIR, f"{task_key if task_key else base_name}_Bayesian_{keep_mode}_260107_SAFE")
        os.makedirs(save_dir, exist_ok=True)

        # 保存 GT 文件 (完全对齐原代码格式)
        df.iloc[self.train_indices].to_csv(os.path.join(save_dir, f"{base_name}_train_GT.csv"), index=False, encoding='utf-8-sig')
        df.iloc[self.test_indices].to_csv(os.path.join(save_dir, f"{base_name}_test_GT.csv"), index=False, encoding='utf-8-sig')

        return df, y_data, missing_mask_new, save_dir, class_labels

    def train_model(self, y_data, missing_mask_new):
        """
        修复版：
        1. 使用训练集的中位数填充缺失值 (Median Imputation)
        2. 手动构建 StandardScaler 时补全 var_ 属性，解决 NotFittedError
        """
        print("\n🚀 开始训练（使用 Median 中位数填充缺失值）...")

        n_rows, n_cols = y_data.shape
        
        # 1. 初始化填充矩阵
        y_data_filled = y_data.copy()
        self._train_medians = [] 

        # 2. 逐列计算训练集中位数，并填充
        for j in range(n_cols):
            col_train_values = y_data[self.train_indices, j]
            col_train_mask = missing_mask_new[self.train_indices, j]
            
            observed_values = col_train_values[~col_train_mask]
            
            if len(observed_values) > 0:
                med_val = float(np.median(observed_values))
            else:
                med_val = 0.0 
            
            self._train_medians.append(med_val)
            y_data_filled[missing_mask_new[:, j], j] = med_val

        # --- 3. 计算统计量并手动“欺骗”Scaler ---

        # 准备用于计算均值方差的数据 (仅用可见位)
        y_train_for_fit = y_data[self.train_indices].copy()
        train_mask = missing_mask_new[self.train_indices]
        y_train_for_fit[train_mask] = np.nan 

        means, scales = [], []
        for j in range(n_cols):
            # 提取非 NaN 数据进行拟合
            obs = y_train_for_fit[:, j][~np.isnan(y_train_for_fit[:, j])].reshape(-1, 1)
            if len(obs) > 0:
                tmp = StandardScaler().fit(obs)
                means.append(tmp.mean_[0])
                scales.append(tmp.scale_[0])
            else:
                means.append(0.0)
                scales.append(1.0)
        
        # 转换为 numpy 数组
        means_arr = np.array(means)
        scales_arr = np.array(scales)
        var_arr = scales_arr ** 2  # <--- 关键修复：计算方差

        # --- 设置 Input Scaler 属性 ---
        self.scaler_Y_input.mean_ = means_arr
        self.scaler_Y_input.scale_ = scales_arr
        self.scaler_Y_input.var_ = var_arr  # <--- 必须赋值
        self.scaler_Y_input.n_features_in_ = n_cols
        self.scaler_Y_input.n_samples_seen_ = np.array([len(self.train_indices)] * n_cols) # 可选，更稳健

        # --- 设置 Target Scaler 属性 ---
        self.scaler_Y_target.mean_ = means_arr
        self.scaler_Y_target.scale_ = scales_arr
        self.scaler_Y_target.var_ = var_arr # <--- 必须赋值
        self.scaler_Y_target.n_features_in_ = n_cols
        self.scaler_Y_target.n_samples_seen_ = np.array([len(self.train_indices)] * n_cols)

        # 4. 执行转换
        X_scaled = self.scaler_Y_input.transform(y_data_filled)
        Y_scaled_target = self.scaler_Y_target.transform(y_data) 

        # 5. 循环训练 BayesianRidge
        final_predictions_scaled = np.zeros_like(X_scaled)

        for j, col_name in enumerate(self.Y_columns):
            target_j = Y_scaled_target[self.train_indices, j]
            
            model = BayesianRidge()
            # 注意：这里的 X 是包含填充值的全量特征
            model.fit(X_scaled[self.train_indices], target_j)

            self.models[col_name] = model
            final_predictions_scaled[:, j] = model.predict(X_scaled)

            if (j + 1) % 5 == 0 or (j + 1) == n_cols:
                print(f"  [进度] 已完成 {j+1}/{n_cols} 列模型的训练")

        self.full_predictions = self.scaler_Y_target.inverse_transform(final_predictions_scaled)
        print("✅ 贝叶斯回归训练及全集预测完成")
        return self.full_predictions

    def perform_evaluation(self, y_true, y_pred, mask, indices, set_name="Test"):
        results = {}
        y_true_sub = y_true[indices]
        y_pred_sub = y_pred[indices]
        mask_sub = mask[indices]

        for j, col in enumerate(self.Y_columns):
            m = mask_sub[:, j]
            if m.sum() > 1:
                t_valid = y_true_sub[m, j]
                p_valid = y_pred_sub[m, j]

                # 剔除无效值
                valid_idx = ~(np.isnan(t_valid) | np.isnan(p_valid))
                if valid_idx.sum() > 1:
                    t, p = t_valid[valid_idx], p_valid[valid_idx]

                    mse = mean_squared_error(t, p)
                    mae = mean_absolute_error(t, p)
                    r2 = r2_score(t, p)
                    corr, p_val = pearsonr(t, p)
                    # 相对误差逻辑
                    rel_err = np.mean(np.abs((p - t) / (np.abs(t) + 1e-8))) * 100

                    results[col] = {
                        'MSE': mse, 'MAE': mae, 'R²': r2,
                        'Correlation': corr, 'P-value': p_val,
                        'Relative_Error_%': rel_err,
                        f'N_{set_name.lower()}_masked': int(m.sum())
                    }

        print(f"\n🎯 {set_name} 评估总结 (仅 Mask 区域):")
        if results:
            res_df = pd.DataFrame(results).T
            print(res_df[['R²', 'Correlation', 'MAE']].round(4))
            return res_df
        return pd.DataFrame()
    def visualize_train_results(self, train_predictions, y_data, missing_mask_new, save_dir="."):
        print("📊 生成训练集可视化图表（仅被 mask 位置）...")
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
                except: r2, corr = 0, 0

                plt.xlabel(f'True {col}')
                plt.ylabel(f'Predicted {col}')
                plt.title(f'{col} (Train - Masked)\nR² = {r2:.3f}, Corr = {corr:.3f}')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'train_set_predictions_masked.png'), dpi=300)
        plt.show()

    def visualize_test_results(self, test_predictions, y_data, missing_mask_new, save_dir="."):
        print("📊 生成测试集可视化图表...")
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
                except: r2, corr = 0, 0

                plt.xlabel(f'True {col}')
                plt.ylabel(f'Predicted {col}')
                plt.title(f'{col} (Test Set)\nR² = {r2:.3f}, Corr = {corr:.3f}')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'test_set_predictions.png'), dpi=300)
        plt.show()

    def plot_training_curve(self, save_dir="."):
        """贝叶斯模型无训练曲线，生成占位图以保持文件一致"""
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Bayesian Ridge: No iterative training curve",
                 ha='center', va='center', fontsize=12)
        plt.savefig(os.path.join(save_dir, 'training_curve.png'), dpi=300)
        plt.close()

    def run_pipeline(self, csv_file_path, smiles_column='psmiles', keep_mode='keep1', task_key=None):
        """完整流程执行器"""
        print("=" * 60)
        print(f"🧪 Bayesian Baseline 预测系统 | 策略: {keep_mode}")
        print("=" * 60)

        # 1. 加载数据
        df, y_data, missing_mask_new, save_dir, class_labels = self.load_and_process_data(
            csv_file_path, smiles_column, keep_mode, task_key
        )
        base_name = _basename_no_ext(csv_file_path)

        # 2. 训练并获得全量预测
        predictions = self.train_model(y_data, missing_mask_new)

        # 3. 评估
        train_res = self.perform_evaluation(y_data, predictions, missing_mask_new, self.train_indices, "Train")
        test_res = self.perform_evaluation(y_data, predictions, missing_mask_new, self.test_indices, "Test")

        # 4. 可视化与保存
        self.visualize_train_results(predictions, y_data, missing_mask_new, save_dir)
        self.visualize_test_results(predictions, y_data, missing_mask_new, save_dir)
        self.plot_training_curve(save_dir)

        # 保存 CSV 结果 (与 126.py 命名完全一致)
        if not train_res.empty:
            train_res.to_csv(os.path.join(save_dir, 'train_set_evaluation_masked.csv'), encoding='utf-8-sig')
        if not test_res.empty:
            test_res.to_csv(os.path.join(save_dir, 'test_set_evaluation.csv'), encoding='utf-8-sig')

        # 5. 性能对比总结
        if not train_res.empty and not test_res.empty:
            comparison = pd.DataFrame({
                'Train_R²': train_res['R²'], 'Test_R²': test_res['R²'],
                'Train_Corr': train_res['Correlation'], 'Test_Corr': test_res['Correlation'],
                'Train_MAE': train_res['MAE'], 'Test_MAE': test_res['MAE']
            })
            comparison.to_csv(os.path.join(save_dir, 'train_test_comparison.csv'), encoding='utf-8-sig')
            print("\n📊 性能对比 (R²):")
            print(comparison[['Train_R²', 'Test_R²']].round(4))

        # 6. 保存详细预测结果
        # 测试集预测
        test_df = df.iloc[self.test_indices].copy()
        test_mask = missing_mask_new[self.test_indices]
        for j, col in enumerate(self.Y_columns):
            vals = test_df[col].values.copy()
            vals[test_mask[:, j]] = predictions[self.test_indices, j][test_mask[:, j]]
            test_df[col] = vals
        test_df.to_csv(os.path.join(save_dir, f"{base_name}_test_predictions.csv"), index=False, encoding='utf-8-sig')

        # 训练集预测
        train_df = df.iloc[self.train_indices].copy()
        train_mask = missing_mask_new[self.train_indices]
        for j, col in enumerate(self.Y_columns):
            vals = train_df[col].values.copy()
            vals[train_mask[:, j]] = predictions[self.train_indices, j][train_mask[:, j]]
            train_df[col] = vals
        train_df.to_csv(os.path.join(save_dir, f"{base_name}_train_predictions.csv"), index=False, encoding='utf-8-sig')

        print(f"\n✅ 流程完成！输出目录: {save_dir}")

# --- 运行示例 (与 126.py 逻辑一致) ---
if __name__ == "__main__":
    predictor = SimpleBayesianYPredictor(random_state=42, test_size=0.2)

    # 任务配置
    data_dict = {'md': 'MD_properties_simple.csv'}
    smiles_col_dict = {'md': 'SMILES'}

    task_key = 'md'
    csv_path = os.path.join(DATA_DIR, data_dict[task_key])
    smiles_col = smiles_col_dict[task_key]

    # 测试不同保留策略
    for km in ['keep1', 'keep2','keep3', 'keep4', 'keep25', 'keep50', 'keep75']:
        predictor.run_pipeline(csv_file_path=csv_path, smiles_column=smiles_col, keep_mode=km, task_key=task_key)


