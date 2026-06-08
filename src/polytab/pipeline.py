# -*- coding: utf-8 -*-
# Auto-generated from ELE-4run_polyBERT_cascade_260107.ipynb

# -*- coding: utf-8 -*-
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get('POLYTAB_ROOT', PACKAGE_DIR.parents[1]))
SCRIPT_DIR = str(PACKAGE_DIR)
DATA_DIR = os.environ.get('POLYTAB_DATA_DIR', str(PROJECT_ROOT / 'data'))
CKPT_DIR = os.environ.get('POLYTAB_OUTPUT_DIR', str(PROJECT_ROOT / 'results'))
POLYBERT_MODEL_PATH = os.environ.get(
    'POLYBERT_MODEL_PATH',
    str(PROJECT_ROOT / 'models' / 'polyBERT')
)

def _configure_plot_fonts():
    """Prefer Arial for Latin glyphs and fall back safely when unavailable."""
    font_candidates = ['Arial', 'Arial Unicode MS', 'Liberation Sans', 'DejaVu Sans']
    local_font_candidates = [
        os.path.join(SCRIPT_DIR, 'Arial.ttf'),
        os.path.join(os.path.dirname(SCRIPT_DIR), 'visualization', 'Arial.ttf'),
        os.path.join(os.path.dirname(SCRIPT_DIR), 'visualization', 'Arial Unicode.ttf')
    ]

    for font_path in local_font_candidates:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    selected_font = next((name for name in font_candidates if name in available_fonts), 'DejaVu Sans')

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [selected_font]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120
    return selected_font

PLOT_FONT_NAME = _configure_plot_fonts()
print(f"Plot font configured: {PLOT_FONT_NAME}")

_original_plt_title = plt.title
def _safe_plt_title(label, *args, **kwargs):
    if isinstance(label, str):
        if '璁粌鎹熷け鏇茬嚎' in label or '训练损失曲线' in label:
            label = 'Training Loss Curve (Train Set Only)'
    return _original_plt_title(label, *args, **kwargs)

plt.title = _safe_plt_title

# BERT相关库
try:
    from transformers import AutoTokenizer, AutoModel
    print("✅ Transformers已导入")
except ImportError:
    print("❌ 需要安装transformers: pip install transformers")
    raise

def _basename_no_ext(path):
    base = os.path.basename(path)
    if '.' in base:
        base = '.'.join(base.split('.')[:-1])
    return base

class BERTFeatureExtractor:
    """polyBERT特征提取器"""
    def __init__(self, model_path, max_length=512):
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """加载polyBERT模型"""
        try:
            print(f"🤗 加载polyBERT模型: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("✅ polyBERT模型加载成功")
        except Exception as e:
            print(f"❌ polyBERT模型加载失败: {e}")
            raise
    
    def extract_cls_token(self, smiles):
        """提取单个SMILES的CLS token特征"""
        if pd.isna(smiles) or smiles == '':
            hidden_size = self.model.config.hidden_size
            return np.zeros(hidden_size, dtype=np.float32)
        
        try:
            smiles_clean = str(smiles).strip()
            inputs = self.tokenizer(
                smiles_clean,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
                add_special_tokens=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            cls_token = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            return cls_token.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ polyBERT特征提取失败: {e}")
            hidden_size = self.model.config.hidden_size
            return np.zeros(hidden_size, dtype=np.float32)
    
    def get_features(self, smiles_list):
        """批量提取polyBERT特征"""
        print("🧬 生成polyBERT分子特征...")
        features = []
        valid_count = 0
        
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                print(f"  处理进度: {i}/{len(smiles_list)}")
            
            feature = self.extract_cls_token(smiles)
            features.append(feature)
            if not np.all(feature == 0):
                valid_count += 1
        
        features = np.array(features, dtype=np.float32)
        print(f"✅ polyBERT特征维度: {features.shape}")
        print(f"✅ 有效分子数: {valid_count}/{len(features)}")
        return features

class MolecularDataset(Dataset):
    """分子数据集类（含分类标签）"""
    def __init__(self, features, targets_Y_scaled, inputs_Y_scaled, missing_mask_Y, class_labels):
        self.features = torch.FloatTensor(features)
        self.targets_Y = torch.FloatTensor(targets_Y_scaled)
        self.inputs_Y = torch.FloatTensor(inputs_Y_scaled)
        self.missing_mask_Y = torch.BoolTensor(missing_mask_Y)
        self.class_labels = torch.LongTensor(class_labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'targets_Y': self.targets_Y[idx],
            'inputs_Y': self.inputs_Y[idx],
            'missing_mask_Y': self.missing_mask_Y[idx],
            'class_labels': self.class_labels[idx]
        }

class SimpleYAttention(nn.Module):
    """简单的Y变量关系注意力机制"""
    def __init__(self, y_dim):
        super(SimpleYAttention, self).__init__()
        self.y_dim = y_dim
        self.attention = nn.Sequential(
            nn.Linear(y_dim, y_dim),
            nn.Tanh(),
            nn.Linear(y_dim, y_dim),
            nn.Softmax(dim=-1)
        )
        self.interaction = nn.Linear(y_dim, y_dim)
        
    def forward(self, y_values):
        attention_weights = self.attention(y_values)
        attended_y = y_values * attention_weights
        enhanced_y = self.interaction(attended_y) + y_values
        return enhanced_y, attention_weights

class CascadeRegressionHead(nn.Module):
    """级联回归头：4层逐步细化"""
    def __init__(self, input_dim, y_dim, hidden_dim=128):
        super().__init__()
        self.y_dim = y_dim
        
        # 第1层回归头（粗略）
        self.reg_head_1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, y_dim)
        )
        
        # 第2层回归头（中等细化）
        self.reg_head_2 = nn.Sequential(
            nn.Linear(input_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, y_dim)
        )
        
        # 第3层回归头（进一步细化）
        self.reg_head_3 = nn.Sequential(
            nn.Linear(input_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, y_dim)
        )
        
        # 第4层回归头（最终精细化）
        self.reg_head_4 = nn.Sequential(
            nn.Linear(input_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, y_dim)
        )
    
    def forward(self, fused_features, class_probs):
        """级联回归：每层在上一层基础上细化"""
        # 第1层：基于分类结果的粗略回归
        pred_1 = self.reg_head_1(fused_features)
        
        # 第2层：融合第1层结果
        fused_1 = torch.cat([fused_features, pred_1], dim=1)
        residual_2 = self.reg_head_2(fused_1)
        pred_2 = pred_1 + residual_2
        
        # 第3层：融合第2层结果
        fused_2 = torch.cat([fused_features, pred_2], dim=1)
        residual_3 = self.reg_head_3(fused_2)
        pred_3 = pred_2 + residual_3
        
        # 第4层：最终精细化
        fused_3 = torch.cat([fused_features, pred_3], dim=1)
        residual_4 = self.reg_head_4(fused_3)
        pred_4 = pred_3 + residual_4
        
        return pred_1, pred_2, pred_3, pred_4

class SimpleSMILESPredictor(nn.Module):
    """带分类头+级联回归头的SMILES-Y预测网络"""
    def __init__(self, feature_dim, y_dim, hidden_dim=256, dropout=0.2, num_classes=5):
        super(SimpleSMILESPredictor, self).__init__()
        self.feature_dim = feature_dim
        self.y_dim = y_dim
        self.num_classes = num_classes
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Y注意力机制
        self.y_attention = SimpleYAttention(y_dim)
        
        # Y编码器
        self.y_encoder = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2.0)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类头：为每个Y预测所属区间
        self.classification_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            ) for _ in range(y_dim)
        ])
        
        # 级联回归头
        self.cascade_regression = CascadeRegressionHead(
            input_dim=hidden_dim * 2,
            y_dim=y_dim,
            hidden_dim=hidden_dim
        )
        
        print(f"📊 模型参数: feature_dim={feature_dim}, y_dim={y_dim}, hidden_dim={hidden_dim}")
        print(f"🔢 分类头: {num_classes}个区间")
        print(f"🎯 回归头: 4层级联细化")
        
    def forward(self, features, y_features, missing_mask):
        # 特征编码
        feature_encoded = self.feature_encoder(features)
        
        # Y注意力增强
        y_enhanced, attention_weights = self.y_attention(y_features)
        y_encoded = self.y_encoder(y_enhanced)
        
        # 特征融合
        fused = torch.cat([feature_encoded, y_encoded], dim=1)
        fused_features = self.fusion(fused)
        
        # 分类头
        class_logits = []
        for i in range(self.y_dim):
            logits = self.classification_head[i](fused_features)
            class_logits.append(logits)
        class_logits = torch.stack(class_logits, dim=1)
        class_probs = torch.softmax(class_logits, dim=-1)
        
        # 级联回归头
        pred_1, pred_2, pred_3, pred_4 = self.cascade_regression(fused_features, class_probs)
        
        return {
            'class_logits': class_logits,
            'class_probs': class_probs,
            'predictions_1': pred_1,
            'predictions_2': pred_2,
            'predictions_3': pred_3,
            'predictions_4': pred_4,
            'attention_weights': attention_weights
        }

class VariationalAutoEncoder(nn.Module):
    """用于Y矩阵的轻量VAE"""
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[256, 128]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 编码器
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(prev, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim)
        
        # 解码器
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def vae_loss_masked(recon, target, mu, logvar, missing_mask, kl_weight=0.1):
    """仅对非缺失位置计算重构MSE"""
    non_missing = (~missing_mask).float()
    mse = ((recon - target) ** 2) * non_missing
    recon_loss = mse.sum() / non_missing.sum().clamp_min(1.0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / recon.shape[0]
    return recon_loss + kl_weight * kl

class SimpleSMILESYPredictor:
    def __init__(self, device=None, random_state=42, alpha_non_missing_loss=0.2,
                 lambda_vae=0.05, beta_vae=0.10, vae_latent_dim=64,
                 bert_model_path=None,
                 lambda_class=1.0, lambda_cascade=[0.8, 0.6, 0.4, 1.0],
                 test_size=0.2):
        """
        参数:
        - test_size: 测试集比例 (默认0.2，即20%)
        - 其他参数保持不变
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        if bert_model_path is None:
            raise ValueError("必须提供polyBERT模型路径 (bert_model_path)")
        
        self.feature_generator = BERTFeatureExtractor(bert_model_path)
        self.feature_dim = None
        
        self.scaler_Y_input = StandardScaler()
        self.scaler_Y_target = StandardScaler()
        
        self.model = None
        self.vae = None
        self.Y_columns = []
        self.low_variance_cols = []
        self.alpha_non_missing_loss = alpha_non_missing_loss
        self.attention_weights = None
        self.lambda_vae = float(lambda_vae)
        self.beta_vae = float(beta_vae)
        self.vae_latent_dim = int(vae_latent_dim)
        
        self.lambda_class = lambda_class
        self.lambda_cascade = lambda_cascade
        
        self.quantile_boundaries = None
        
        # 新增：训练集/测试集索引
        self.test_size = test_size
        self.train_indices = None
        self.test_indices = None
        self._train_medians = None  # 存储训练集中位数
        
        print(f"🖥️ 使用设备: {self.device}")
        print(f"🔧 VAE设置: lambda_vae={self.lambda_vae}, beta_vae={self.beta_vae}, latent={self.vae_latent_dim}")
        print(f"🎯 损失权重: lambda_class={lambda_class}, lambda_cascade={lambda_cascade}")
        print(f"🔀 数据集划分: 训练集={1-test_size:.0%}, 测试集={test_size:.0%}")
        print(f"🧬 使用polyBERT特征提取")
    
    def _build_keep_mask(self, n_rows, n_cols, keep_k, rng):
        """为每一行保留 keep_k 个列，其余置为缺失"""
        keep_k = int(max(1, min(keep_k, n_cols - 1)))
        missing_mask_new = np.ones((n_rows, n_cols), dtype=bool)
        for i in range(n_rows):
            keep_cols = rng.choice(n_cols, size=keep_k, replace=False)
            missing_mask_new[i, keep_cols] = False
        return missing_mask_new, keep_k

    def _compute_quantile_boundaries(self, Y_data, missing_mask_new):
        """计算每列的四分位数边界（仅用保留数据）"""
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
        
        print("\n📊 四分位数边界（仅用训练集保留数据计算）:")
        for col, bounds in quantile_boundaries.items():
            print(f"  {col}: min={bounds[0]:.3f}, Q1={bounds[1]:.3f}, Q2={bounds[2]:.3f}, "
                  f"Q3={bounds[3]:.3f}, max={bounds[4]:.3f}")
        
        return quantile_boundaries
    
    def _assign_class_labels(self, Y_data, missing_mask_new):
        """为每个样本的每个Y分配区间类别（0-4）"""
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
                elif val <= bounds[4]:
                    class_labels[i, j] = 3
                else:
                    class_labels[i, j] = 4
        
        return class_labels

    def load_and_process_data(self, csv_file_path, smiles_column='psmiles',
                              keep_mode='keep1', save_files=True, task_key=None):
        """加载与处理数据（严格训练/测试集分离）"""
        print(f"📂 加载数据: {csv_file_path}")
        print("🔒 严格模式：训练集与测试集完全分离，无数据泄漏")

        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        df_raw = None
        for encoding in encodings:
            try:
                df_raw = pd.read_csv(csv_file_path, encoding=encoding)
                print(f"✅ 读取成功 (编码: {encoding})")
                break
            except UnicodeDecodeError:
                continue
        if df_raw is None:
            raise ValueError("无法读取CSV文件")

        unnamed_cols = [c for c in df_raw.columns if str(c).startswith('Unnamed:')]
        if unnamed_cols:
            print(f"🧹 移除未命名列: {unnamed_cols}")
            df_raw = df_raw.drop(columns=unnamed_cols)

        print(f"📊 原始数据: {df_raw.shape}")
        if smiles_column not in df_raw.columns:
            raise ValueError(f"未找到SMILES列: {smiles_column}")

        # 识别候选Y列
        candidate_cols = []
        for col in df_raw.columns:
            if col == smiles_column:
                continue
            s = pd.to_numeric(df_raw[col], errors='coerce')
            if not s.isnull().all():
                candidate_cols.append(col)

        # 删除含NaN的行
        y_df_numeric = df_raw[candidate_cols].apply(pd.to_numeric, errors='coerce')
        non_na_mask = ~y_df_numeric.isnull().any(axis=1)
        df = df_raw.loc[non_na_mask].reset_index(drop=True)
        y_df_numeric = df[candidate_cols].apply(pd.to_numeric, errors='coerce')
        print(f"🧹 删除含NaN样本后数据: {df.shape}")

        # 统计与低方差过滤
        stats = []
        for col in candidate_cols:
            vals = pd.to_numeric(df[col], errors='coerce').values
            std = float(np.std(vals)) if len(vals) > 0 else 0.0
            uniq = int(len(np.unique(vals))) if len(vals) > 0 else 0
            stats.append((col, std, uniq))
        self.low_variance_cols = [c for c, std, uniq in stats if std < 1e-8 or uniq <= 1]
        self.Y_columns = [c for c, std, uniq in stats if c not in self.low_variance_cols]
        
        print("🔬 列统计:")
        for c, std, uniq in stats:
            tag = "LOW-VAR" if c in self.low_variance_cols else ""
            print(f"  - {c}: std={std:.3e}, nunique={uniq} {tag}")
        print(f"✅ 有效Y列数量: {len(self.Y_columns)}")
        
        if len(self.Y_columns) == 0:
            raise ValueError("有效的Y列为空")

        # === 【关键修改1】先划分训练/测试集 ===
        n_samples = len(df)
        all_indices = np.arange(n_samples)
        
        self.train_indices, self.test_indices = train_test_split(
            all_indices,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )
        
        print(f"\n🔀 数据集划分:")
        print(f"   训练集: {len(self.train_indices)} 样本 ({(1-self.test_size)*100:.0f}%)")
        print(f"   测试集: {len(self.test_indices)} 样本 ({self.test_size*100:.0f}%)")
        print("   ⚠️  测试集将完全不参与训练和标准化过程\n")

        # polyBERT特征提取
        smiles_list = df[smiles_column].tolist()
        features = self.feature_generator.get_features(smiles_list)
        
        if len(features) > 0:
            self.feature_dim = features.shape[1]
            print(f"✅ 动态获取polyBERT特征维度: {self.feature_dim}")

        y_data = df[self.Y_columns].values.astype(np.float32)
        n_rows, n_cols = y_data.shape

        # 决定保留比例
        if keep_mode == 'keep1':
            keep_k = 1
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
        print(f"📌 保留策略: {keep_mode} -> 每行保留 {keep_k}/{n_cols} 个值")

        # === 【关键修改2】仅用训练集计算四分位数边界 ===
        y_data_train = y_data[self.train_indices]
        missing_mask_train = missing_mask_new[self.train_indices]
        
        self._compute_quantile_boundaries(y_data_train, missing_mask_train)
        
        # 分配分类标签（全部数据，但边界仅来自训练集）
        class_labels = self._assign_class_labels(y_data, missing_mask_new)
        print(f"✅ 已生成分类标签: {class_labels.shape}")

        # 构造训练输入版本（未保留位置写NaN）
        df_missing_for_training = df.copy()
        y_data_missing = y_data.copy().astype(object)
        y_data_missing[missing_mask_new] = np.nan
        for j, col in enumerate(self.Y_columns):
            df_missing_for_training[col] = y_data_missing[:, j]

        # 保存目录
        base_name = _basename_no_ext(csv_file_path)
        task_key = task_key if task_key else base_name
        save_dir = os.path.join(CKPT_DIR, f"{task_key}_polyBERT_cascade_{keep_mode}_260107_SAFE")
        os.makedirs(save_dir, exist_ok=True)
        print(f"🗂️ 保存目录: {save_dir}")

        # 分别保存训练集和测试集真值
        df_train = df.iloc[self.train_indices].reset_index(drop=True)
        df_test = df.iloc[self.test_indices].reset_index(drop=True)
        
        train_gt_path = os.path.join(save_dir, f"{base_name}_train_GT.csv")
        test_gt_path = os.path.join(save_dir, f"{base_name}_test_GT.csv")
        
        df_train.to_csv(train_gt_path, index=False, encoding='utf-8-sig')
        df_test.to_csv(test_gt_path, index=False, encoding='utf-8-sig')
        
        print(f"💾 已保存训练集真值: {train_gt_path}")
        print(f"💾 已保存测试集真值: {test_gt_path}")

        # 保存带缺失的版本
        df_missing_to_save = df.copy().astype(object)
        for j, col in enumerate(self.Y_columns):
            col_vals = df_missing_to_save[col].values
            write_empty = missing_mask_new[:, j]
            col_vals[write_empty] = ""
            df_missing_to_save[col] = col_vals

        missing_path = os.path.join(save_dir, f"{base_name}_{keep_mode}.csv")
        df_missing_to_save.to_csv(missing_path, index=False, encoding='utf-8-sig')
        print(f"💾 已保存保留策略CSV: {missing_path}")

        total_missing = missing_mask_new.sum()
        overall_rate = total_missing / (n_rows * n_cols) * 100.0
        print(f"📊 置空比例: {overall_rate:.2f}%")

        original_nan_mask = np.zeros_like(missing_mask_new, dtype=bool)

        return df, df_missing_for_training, features, missing_mask_new, original_nan_mask, save_dir, class_labels
    
    def train_model(self, features, df_complete, df_missing, missing_mask_new, class_labels,
                    epochs=200, batch_size=32, lr=0.001):
        """训练模型（仅用训练集，无数据泄漏）"""
        print("\n🚀 开始训练模型（仅使用训练集，测试集完全隔离）...")
        print("⚠️  关键：标准化器、四分位数等统计量仅在训练集上计算")
        
        # === 【关键修改】仅使用训练集数据 ===
        train_features = features[self.train_indices]
        train_Y_complete = df_complete.iloc[self.train_indices][self.Y_columns].values.astype(np.float32)
        train_Y_missing = df_missing.iloc[self.train_indices][self.Y_columns].values.astype(np.float32)
        train_missing_mask = missing_mask_new[self.train_indices]
        train_class_labels = class_labels[self.train_indices]
        
        print(f"📊 训练集规模: {len(self.train_indices)} 样本")
        print(f"🔒 测试集规模: {len(self.test_indices)} 样本（完全未使用）")
        
        missing_mask = np.isnan(train_Y_missing) | np.isnan(train_Y_complete)

        # 填充缺失值（仅用训练集保留数据计算中位数）
        Y_data_filled = train_Y_missing.copy()
        self._train_medians = []  # 保存训练集中位数，供测试集使用
        
        for i in range(Y_data_filled.shape[1]):
            col_values = Y_data_filled[:, i]
            observed_mask = ~train_missing_mask[:, i]
            observed_values = col_values[observed_mask]
            
            if len(observed_values) > 0:
                col_median = np.nanmedian(observed_values)
            else:
                col_median = 0.0
            
            if np.isnan(col_median):
                col_median = 0.0
            
            self._train_medians.append(col_median)
            Y_data_filled[np.isnan(Y_data_filled[:, i]), i] = col_median
        
        print(f"✅ 已保存训练集中位数: {self._train_medians[:3]}...")
        
        # === 【关键修改】标准化器仅在训练集保留数据上训练 ===
        print("📊 标准化数据（仅使用训练集保留数据）...")
        
        # 输入侧标准化
        Y_for_input_scaler = Y_data_filled.copy()
        Y_for_input_scaler_masked = Y_for_input_scaler.copy()
        Y_for_input_scaler_masked[train_missing_mask] = np.nan
        
        self.scaler_Y_input = StandardScaler()
        n_cols = Y_for_input_scaler_masked.shape[1]
        
        means = []
        scales = []
        
        for j in range(n_cols):
            col_data = Y_for_input_scaler_masked[:, j]
            observed_values = col_data[~np.isnan(col_data)].reshape(-1, 1)
            
            if len(observed_values) > 0:
                temp_scaler = StandardScaler()
                temp_scaler.fit(observed_values)
                means.append(temp_scaler.mean_[0])
                scales.append(temp_scaler.scale_[0])
            else:
                means.append(0.0)
                scales.append(1.0)
        
        self.scaler_Y_input.mean_ = np.array(means)
        self.scaler_Y_input.scale_ = np.array(scales)
        self.scaler_Y_input.n_features_in_ = n_cols
        self.scaler_Y_input.n_samples_seen_ = Y_for_input_scaler.shape[0]
        
        Y_scaled_input = self.scaler_Y_input.transform(Y_data_filled)
        
        # 目标侧标准化
        Y_complete_for_scaler = train_Y_complete.copy()
        Y_complete_for_scaler_masked = Y_complete_for_scaler.copy()
        Y_complete_for_scaler_masked[train_missing_mask] = np.nan
        
        for j in range(Y_complete_for_scaler.shape[1]):
            col = Y_complete_for_scaler[:, j]
            observed_mask = ~train_missing_mask[:, j]
            observed_values = col[observed_mask]
            
            if len(observed_values) > 0:
                med = np.nanmedian(observed_values)
            else:
                med = 0.0
            
            if np.isnan(med):
                med = 0.0
            col[np.isnan(col)] = med
            Y_complete_for_scaler[:, j] = col
        
        self.scaler_Y_target = StandardScaler()
        means_target = []
        scales_target = []
        
        for j in range(n_cols):
            col_data = Y_complete_for_scaler_masked[:, j]
            observed_values = col_data[~np.isnan(col_data)].reshape(-1, 1)
            
            if len(observed_values) > 0:
                temp_scaler = StandardScaler()
                temp_scaler.fit(observed_values)
                means_target.append(temp_scaler.mean_[0])
                scales_target.append(temp_scaler.scale_[0])
            else:
                means_target.append(0.0)
                scales_target.append(1.0)
        
        self.scaler_Y_target.mean_ = np.array(means_target)
        self.scaler_Y_target.scale_ = np.array(scales_target)
        self.scaler_Y_target.n_features_in_ = n_cols
        self.scaler_Y_target.n_samples_seen_ = Y_complete_for_scaler.shape[0]
        
        Y_scaled_target = self.scaler_Y_target.transform(Y_complete_for_scaler)
        
        print(f"✅ 输入scaler统计(仅训练集): mean={self.scaler_Y_input.mean_[:3]}, scale={self.scaler_Y_input.scale_[:3]}")
        print(f"✅ 目标scaler统计(仅训练集): mean={self.scaler_Y_target.mean_[:3]}, scale={self.scaler_Y_target.scale_[:3]}")
        
        dataset = MolecularDataset(train_features, Y_scaled_target, Y_scaled_input, missing_mask, train_class_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 构建模型
        self.model = SimpleSMILESPredictor(
            feature_dim=self.feature_dim,
            y_dim=len(self.Y_columns),
            hidden_dim=256,
            dropout=0.2,
            num_classes=5
        ).to(self.device)

        self.vae = VariationalAutoEncoder(
            input_dim=len(self.Y_columns),
            latent_dim=min(self.vae_latent_dim, max(8, len(self.Y_columns)//2)),
            hidden_dims=[256, 128, 64]
        ).to(self.device)
        
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.vae.parameters()),
            lr=lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=15, min_lr=1e-6
        )
        
        self.model.train()
        self.vae.train()
        losses = []
        best_loss = float('inf')
        patience = 0
        
        print(f"🏋️ 训练参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
        print(f"🎯 损失权重: lambda_class={self.lambda_class}, lambda_cascade={self.lambda_cascade}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_class_loss = 0.0
            epoch_reg_losses = [0.0, 0.0, 0.0, 0.0]
            batch_count = 0
            
            for batch in dataloader:
                feature_batch = batch['features'].to(self.device)
                y_target = batch['targets_Y'].to(self.device)
                y_input = batch['inputs_Y'].to(self.device)
                missing_mask_batch = batch['missing_mask_Y'].to(self.device)
                class_labels_batch = batch['class_labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(feature_batch, y_input, missing_mask_batch)
                
                # 分类损失
                class_logits = outputs['class_logits']
                class_loss = 0.0
                for i in range(len(self.Y_columns)):
                    class_loss += nn.CrossEntropyLoss()(
                        class_logits[:, i, :],
                        class_labels_batch[:, i]
                    )
                class_loss = class_loss / len(self.Y_columns)
                
                # 级联回归损失
                reg_losses = []
                for layer_idx, pred_key in enumerate(['predictions_1', 'predictions_2', 
                                                      'predictions_3', 'predictions_4']):
                    pred = outputs[pred_key]
                    reg_loss = self.compute_loss_weighted(
                        pred, y_target, missing_mask_batch, alpha=self.alpha_non_missing_loss
                    )
                    reg_losses.append(reg_loss * self.lambda_cascade[layer_idx])
                
                # VAE损失
                vae_recon, mu, logvar = self.vae(y_input)
                loss_vae = vae_loss_masked(vae_recon, y_input, mu, logvar, missing_mask_batch, kl_weight=0.1)
                
                # 总损失
                loss = (self.lambda_class * class_loss + 
                        sum(reg_losses) + 
                        self.lambda_vae * loss_vae)
                
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.vae.parameters()), 
                        max_norm=1.0
                    )
                    optimizer.step()
                    epoch_loss += loss.item()
                    epoch_class_loss += class_loss.item()
                    for i, rl in enumerate(reg_losses):
                        epoch_reg_losses[i] += rl.item()
                    batch_count += 1

            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                avg_class_loss = epoch_class_loss / batch_count
                avg_reg_losses = [rl / batch_count for rl in epoch_reg_losses]
                losses.append(avg_loss)
                scheduler.step(avg_loss)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience = 0
                else:
                    patience += 1
                
                if (epoch + 1) % 25 == 0:
                    print(f"📈 Epoch {epoch+1}/{epochs}")
                    print(f"   总损失: {avg_loss:.6f}")
                    print(f"   分类损失: {avg_class_loss:.6f}")
                    print(f"   回归损失: L1={avg_reg_losses[0]:.6f}, L2={avg_reg_losses[1]:.6f}, "
                          f"L3={avg_reg_losses[2]:.6f}, L4={avg_reg_losses[3]:.6f}")
                    print(f"   LR: {optimizer.param_groups[0]['lr']:.8f}")
                
                if patience >= 30:
                    print(f"🛑 早停在epoch {epoch+1}")
                    break
        
        print("✅ 训练完成!")
        return losses
    
    def compute_loss_weighted(self, predictions, targets, missing_mask, alpha=0.2):
        """加权MSE"""
        w = torch.where(missing_mask, torch.ones_like(targets), torch.full_like(targets, alpha))
        mse = (w * (predictions - targets) ** 2).sum() / w.sum().clamp_min(1.0)
        return mse

    def _vae_reconstruct(self, Y_scaled_input):
        """VAE重构"""
        assert self.vae is not None, "VAE未训练"
        self.vae.eval()
        with torch.no_grad():
            y_tensor = torch.FloatTensor(Y_scaled_input).to(self.device)
            recon_scaled, _, _ = self.vae(y_tensor)
            recon_scaled = recon_scaled.cpu().numpy()
        recon_values = self.scaler_Y_input.inverse_transform(recon_scaled)
        return recon_values

    def evaluate_on_train_set(self, features, df_complete, df_missing, missing_mask_new):
        """在训练集上评估（仅评估被mask的位置，无泄漏）"""
        print("\n📊 在训练集上评估（仅评估训练时被mask的位置）...")
        print("⚠️  评估位置：训练过程中未见过的缺失位置（无泄漏）")
    
        if self.model is None:
            raise ValueError("需要先训练模型")
    
        # === 仅使用训练集数据 ===
        train_features = features[self.train_indices]
        train_Y_complete = df_complete.iloc[self.train_indices][self.Y_columns].values.astype(np.float32)
        train_Y_missing = df_missing.iloc[self.train_indices][self.Y_columns].values.astype(np.float32)
        train_missing_mask = missing_mask_new[self.train_indices]
    
        print(f"🔬 训练集样本数: {len(self.train_indices)}")
        print(f"📌 评估策略: 仅在被mask的位置计算误差（模型训练时未见过真值）")
    
        # 填充训练集缺失值
        Y_data_filled = train_Y_missing.copy()
        for i in range(Y_data_filled.shape[1]):
            fill_value = self._train_medians[i]
            Y_data_filled[np.isnan(Y_data_filled[:, i]), i] = fill_value
    
        # 标准化
        Y_scaled_input = self.scaler_Y_input.transform(Y_data_filled)
    
        # 预测
        self.model.eval()
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(train_features).to(self.device)
            y_tensor = torch.FloatTensor(Y_scaled_input).to(self.device)
            dummy_mask = torch.zeros_like(y_tensor, dtype=torch.bool).to(self.device)
        
            outputs = self.model(feature_tensor, y_tensor, dummy_mask)
            predictions = outputs['predictions_4']
            main_pred = self.scaler_Y_target.inverse_transform(predictions.cpu().numpy())
    
        # VAE弱融合
        if self.vae is not None and self.beta_vae > 0:
            vae_recon = self._vae_reconstruct(Y_scaled_input)
            fused = (1.0 - self.beta_vae) * main_pred + self.beta_vae * vae_recon
        else:
            fused = main_pred
    
        # 【关键】仅在被mask的位置评估（训练时未见过真值）
        results = {}
        for i, col in enumerate(self.Y_columns):
            mask = train_missing_mask[:, i]  # 训练时被mask的位置
            if mask.sum() > 1:
                pred_missing = fused[mask, i]
                true_missing = train_Y_complete[mask, i]
            
                valid_idx = ~(np.isnan(pred_missing) | np.isnan(true_missing))
                if valid_idx.sum() > 1:
                    pred_valid = pred_missing[valid_idx]
                    true_valid = true_missing[valid_idx]
                
                    mse = mean_squared_error(true_valid, pred_valid)
                    mae = mean_absolute_error(true_valid, pred_valid)
                    r2 = r2_score(true_valid, pred_valid)
                    corr, p_val = pearsonr(true_valid, pred_valid)
                    relative_error = np.mean(np.abs((pred_valid - true_valid) /
                                                (np.abs(true_valid) + 1e-8))) * 100
                
                    results[col] = {
                        'MSE': mse,
                        'MAE': mae,
                        'R²': r2,
                        'Correlation': corr,
                        'P-value': p_val,
                        'Relative_Error_%': relative_error,
                        'N_train_masked': int(mask.sum()),
                        'N_valid': int(valid_idx.sum())
                    }
    
        print("\n🎯 训练集评估结果（仅被mask位置）:")
        if results:
            results_df = pd.DataFrame(results).T
            print(results_df[['R²', 'Correlation', 'MAE', 'N_train_masked']].round(4))
            print(f"\n平均R²: {results_df['R²'].mean():.4f}")
            print(f"平均相关系数: {results_df['Correlation'].mean():.4f}")
            print(f"平均相对误差: {results_df['Relative_Error_%'].mean():.2f}%")
    
        return results, fused

    def visualize_train_results(self, train_predictions, df_complete, missing_mask_new, save_dir="."):
        """可视化训练集结果（仅被mask位置）"""
        print("📊 生成训练集可视化图表（仅被mask位置）...")
    
        train_Y_complete = df_complete.iloc[self.train_indices][self.Y_columns].values.astype(np.float32)
        train_missing_mask = missing_mask_new[self.train_indices]
    
        n_cols = 3
        n_rows = min(3, (len(self.Y_columns) + n_cols - 1) // n_cols)
        n_show = min(9, len(self.Y_columns))
    
        plt.figure(figsize=(15, 5 * n_rows))
    
        for i in range(n_show):
            col = self.Y_columns[i]
            plt.subplot(n_rows, n_cols, i + 1)
        
            mask = train_missing_mask[:, i]
            if mask.sum() > 0:
                true_missing = train_Y_complete[mask, i]
                pred_missing = train_predictions[mask, i]
            
                valid_idx = ~(np.isnan(pred_missing) | np.isnan(true_missing))
                if valid_idx.sum() > 0:
                    true_valid = true_missing[valid_idx]
                    pred_valid = pred_missing[valid_idx]
                
                    plt.scatter(true_valid, pred_valid, alpha=0.6, s=30, color='blue')
                
                    min_val = min(true_valid.min(), pred_valid.min())
                    max_val = max(true_valid.max(), pred_valid.max())
                    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
                
                    try:
                        r2 = r2_score(true_valid, pred_valid)
                        corr, _ = pearsonr(true_valid, pred_valid)
                    except:
                        r2, corr = 0, 0
                
                    plt.xlabel(f'True {col}')
                    plt.ylabel(f'Predicted {col}')
                    plt.title(f'{col} (Train Set - Masked)\nR² = {r2:.3f}, Corr = {corr:.3f}')
                    plt.text(0.05, 0.95, f'N = {len(pred_valid)}', 
                             transform=plt.gca().transAxes,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
        plt.tight_layout()
        fig_path = os.path.join(save_dir, 'train_set_predictions_masked.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 训练集图已保存: {fig_path}")
    
    def evaluate_on_test_set(self, features, df_complete, df_missing, missing_mask_new):
        """在测试集上评估（完全未见过的数据）"""
        print("\n📊 在测试集上评估（完全未见过的数据）...")
        
        if self.model is None:
            raise ValueError("需要先训练模型")
        
        # === 仅使用测试集数据 ===
        test_features = features[self.test_indices]
        test_Y_complete = df_complete.iloc[self.test_indices][self.Y_columns].values.astype(np.float32)
        test_Y_missing = df_missing.iloc[self.test_indices][self.Y_columns].values.astype(np.float32)
        test_missing_mask = missing_mask_new[self.test_indices]
        
        print(f"🔬 测试集样本数: {len(self.test_indices)}")
        
        # 填充测试集缺失值（使用训练集学到的中位数）
        Y_data_filled = test_Y_missing.copy()
        for i in range(Y_data_filled.shape[1]):
            col_values = Y_data_filled[:, i]
            fill_value = self._train_medians[i]  # 使用训练集中位数
            Y_data_filled[np.isnan(Y_data_filled[:, i]), i] = fill_value
        
        # 使用训练好的scaler（已经在训练集上fit过）
        Y_scaled_input = self.scaler_Y_input.transform(Y_data_filled)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(test_features).to(self.device)
            y_tensor = torch.FloatTensor(Y_scaled_input).to(self.device)
            dummy_mask = torch.zeros_like(y_tensor, dtype=torch.bool).to(self.device)
            
            outputs = self.model(feature_tensor, y_tensor, dummy_mask)
            predictions = outputs['predictions_4']
            main_pred = self.scaler_Y_target.inverse_transform(predictions.cpu().numpy())
            
            # 保存注意力权重和分类概率
            self.attention_weights = outputs['attention_weights'].cpu().numpy()
            self.class_probs = outputs['class_probs'].cpu().numpy()
        
        # VAE弱融合
        if self.vae is not None and self.beta_vae > 0:
            vae_recon = self._vae_reconstruct(Y_scaled_input)
            fused = (1.0 - self.beta_vae) * main_pred + self.beta_vae * vae_recon
        else:
            fused = main_pred
        
        # 计算指标（仅在置空位置）
        results = {}
        for i, col in enumerate(self.Y_columns):
            mask = test_missing_mask[:, i]
            if mask.sum() > 1:
                pred_missing = fused[mask, i]
                true_missing = test_Y_complete[mask, i]
                
                valid_idx = ~(np.isnan(pred_missing) | np.isnan(true_missing))
                if valid_idx.sum() > 1:
                    pred_valid = pred_missing[valid_idx]
                    true_valid = true_missing[valid_idx]
                    
                    mse = mean_squared_error(true_valid, pred_valid)
                    mae = mean_absolute_error(true_valid, pred_valid)
                    r2 = r2_score(true_valid, pred_valid)
                    corr, p_val = pearsonr(true_valid, pred_valid)
                    relative_error = np.mean(np.abs((pred_valid - true_valid) /
                                                    (np.abs(true_valid) + 1e-8))) * 100
                    
                    results[col] = {
                        'MSE': mse,
                        'MAE': mae,
                        'R²': r2,
                        'Correlation': corr,
                        'P-value': p_val,
                        'Relative_Error_%': relative_error,
                        'N_test_missing': int(mask.sum()),
                        'N_valid': int(valid_idx.sum())
                    }
        
        print("\n🎯 测试集评估结果:")
        if results:
            results_df = pd.DataFrame(results).T
            print(results_df[['R²', 'Correlation', 'MAE', 'N_test_missing']].round(4))
            print(f"\n平均R²: {results_df['R²'].mean():.4f}")
            print(f"平均相关系数: {results_df['Correlation'].mean():.4f}")
            print(f"平均相对误差: {results_df['Relative_Error_%'].mean():.2f}%")
        
        return results, fused
    
    def visualize_test_results(self, test_predictions, df_complete, missing_mask_new, save_dir="."):
        """可视化测试集结果"""
        print("📊 生成测试集可视化图表...")
        
        test_Y_complete = df_complete.iloc[self.test_indices][self.Y_columns].values.astype(np.float32)
        test_missing_mask = missing_mask_new[self.test_indices]
        
        n_cols = 3
        n_rows = min(3, (len(self.Y_columns) + n_cols - 1) // n_cols)
        n_show = min(9, len(self.Y_columns))
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i in range(n_show):
            col = self.Y_columns[i]
            plt.subplot(n_rows, n_cols, i + 1)
            
            mask = test_missing_mask[:, i]
            if mask.sum() > 0:
                true_missing = test_Y_complete[mask, i]
                pred_missing = test_predictions[mask, i]
                
                valid_idx = ~(np.isnan(pred_missing) | np.isnan(true_missing))
                if valid_idx.sum() > 0:
                    true_valid = true_missing[valid_idx]
                    pred_valid = pred_missing[valid_idx]
                    
                    plt.scatter(true_valid, pred_valid, alpha=0.6, s=30, color='red')
                    
                    min_val = min(true_valid.min(), pred_valid.min())
                    max_val = max(true_valid.max(), pred_valid.max())
                    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
                    
                    try:
                        r2 = r2_score(true_valid, pred_valid)
                        corr, _ = pearsonr(true_valid, pred_valid)
                    except:
                        r2, corr = 0, 0
                    
                    plt.xlabel(f'True {col}')
                    plt.ylabel(f'Predicted {col}')
                    plt.title(f'{col} (Test Set)\nR² = {r2:.3f}, Corr = {corr:.3f}')
                    plt.text(0.05, 0.95, f'N = {len(pred_valid)}', 
                             transform=plt.gca().transAxes,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        fig_path = os.path.join(save_dir, 'test_set_predictions.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 测试集图已保存: {fig_path}")

    def plot_training_curve(self, losses, save_dir="."):
        if losses:
            plt.figure(figsize=(10, 6))
            plt.plot(losses, 'b-', linewidth=2, label='Total Loss')
            plt.title('训练损失曲线 (仅训练集)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            fig_path = os.path.join(save_dir, 'training_curve.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"💾 曲线已保存: {fig_path}")

    def run_pipeline(self, csv_file_path, smiles_column='psmiles',
                     keep_mode='keep1', task_key=None):
        """完整流程（严格训练/测试集分离版本）"""
        print("=" * 60)
        print("🧪 polyBERT-Cascade预测系统（无数据泄漏版本）")
        print("🔬 严格训练/测试集分离 + 标准化无泄漏")
        print("=" * 60)
        
        try:
            # 1) 数据加载（含训练/测试集划分）
            (df_complete, df_missing_for_training, features,
             missing_mask_new, original_nan_mask, save_dir, class_labels) = self.load_and_process_data(
                csv_file_path=csv_file_path,
                smiles_column=smiles_column,
                keep_mode=keep_mode,
                save_files=True,
                task_key=task_key
            )
            base_name = _basename_no_ext(csv_file_path)

            # 2) 训练模型（仅用训练集）
            losses = self.train_model(
                features, df_complete, df_missing_for_training, 
                missing_mask_new, class_labels, epochs=500, batch_size=16
            )

            # 3a) 在训练集上评估（仅被mask位置，无泄漏）
            print("\n" + "="*60)
            print("📊 评估阶段1：训练集性能（仅被mask位置）")
            print("="*60)
            train_results, train_predictions = self.evaluate_on_train_set(
                features, df_complete, df_missing_for_training, missing_mask_new
            )
            
            # 可视化训练集结果
            self.visualize_train_results(train_predictions, df_complete, missing_mask_new, save_dir=save_dir)
            
            # 保存训练集评估结果
            if train_results:
                train_results_df = pd.DataFrame(train_results).T
                train_eval_path = os.path.join(save_dir, 'train_set_evaluation_masked.csv')
                train_results_df.to_csv(train_eval_path, encoding='utf-8-sig')
                print(f"💾 训练集评估结果已保存: {train_eval_path}")

            # 3b) 在测试集上评估
            print("\n" + "="*60)
            print("📊 评估阶段2：测试集性能（完全未见过的数据）")
            print("="*60)
            test_results, test_predictions = self.evaluate_on_test_set(
                features, df_complete, df_missing_for_training, missing_mask_new
            )

            # 4) 可视化测试集结果
            self.visualize_test_results(test_predictions, df_complete, missing_mask_new, save_dir=save_dir)
            self.plot_training_curve(losses, save_dir=save_dir)

            # 5) 对比训练集和测试集性能
            print("\n" + "="*60)
            print("📊 性能对比总结 (Train vs Test)")
            print("="*60)
            
            if train_results and test_results:
                train_df = pd.DataFrame(train_results).T
                test_df = pd.DataFrame(test_results).T
                
                comparison = pd.DataFrame({
                    'Train_R²': train_df['R²'],
                    'Test_R²': test_df['R²'],
                    'Train_Corr': train_df['Correlation'],
                    'Test_Corr': test_df['Correlation'],
                    'Train_MAE': train_df['MAE'],
                    'Test_MAE': test_df['MAE'],
                    'Train_N': train_df['N_train_masked'],
                    'Test_N': test_df['N_test_missing']
                })
                comparison['R²_Gap'] = comparison['Train_R²'] - comparison['Test_R²']
                comparison['Corr_Gap'] = comparison['Train_Corr'] - comparison['Test_Corr']
                
                print("\n📊 详细对比:")
                print(comparison.round(4))
                
                print(f"\n🎯 总体统计:")
                print(f"   训练集平均R²: {comparison['Train_R²'].mean():.4f}")
                print(f"   测试集平均R²: {comparison['Test_R²'].mean():.4f}")
                print(f"   R²差距 (Train-Test): {comparison['R²_Gap'].mean():.4f}")
                print(f"\n   训练集平均相关系数: {comparison['Train_Corr'].mean():.4f}")
                print(f"   测试集平均相关系数: {comparison['Test_Corr'].mean():.4f}")
                print(f"   相关系数差距: {comparison['Corr_Gap'].mean():.4f}")
                print(f"\n   训练集平均MAE: {comparison['Train_MAE'].mean():.4f}")
                print(f"   测试集平均MAE: {comparison['Test_MAE'].mean():.4f}")
                
                # 判断过拟合程度
                avg_gap = comparison['R²_Gap'].mean()
                if avg_gap < 0.05:
                    print(f"\n✅ 过拟合程度: 很低 (R²差距={avg_gap:.4f})")
                elif avg_gap < 0.10:
                    print(f"\n⚠️  过拟合程度: 轻微 (R²差距={avg_gap:.4f})")
                else:
                    print(f"\n❌ 过拟合程度: 较高 (R²差距={avg_gap:.4f})")
                
                # 保存对比结果
                comp_path = os.path.join(save_dir, 'train_test_comparison.csv')
                comparison.to_csv(comp_path, encoding='utf-8-sig')
                print(f"\n💾 对比结果已保存: {comp_path}")

            # 6) 保存测试集评估结果
            if test_results:
                test_results_df = pd.DataFrame(test_results).T
                test_eval_path = os.path.join(save_dir, 'test_set_evaluation.csv')
                test_results_df.to_csv(test_eval_path, encoding='utf-8-sig')
                print(f"💾 测试集评估结果已保存: {test_eval_path}")
                
                print("\n📊 测试集性能总结:")
                print(test_results_df.round(4))
                print(f"\n🎯 测试集总体性能:")
                print(f"   平均R²: {test_results_df['R²'].mean():.3f}")
                print(f"   平均相关系数: {test_results_df['Correlation'].mean():.3f}")
                print(f"   平均相对误差: {test_results_df['Relative_Error_%'].mean():.1f}%")

            # 7) 保存测试集预测结果
            test_pred_df = df_complete.iloc[self.test_indices].copy()
            test_missing_mask = missing_mask_new[self.test_indices]
            
            for j, col in enumerate(self.Y_columns):
                col_vals = test_pred_df[col].values.copy()
                col_vals[test_missing_mask[:, j]] = test_predictions[test_missing_mask[:, j], j]
                test_pred_df[col] = col_vals
            
            test_pred_path = os.path.join(save_dir, f"{base_name}_test_predictions.csv")
            test_pred_df.to_csv(test_pred_path, index=False, encoding='utf-8-sig')
            print(f"💾 已保存测试集预测: {test_pred_path}")
            
            # 7b) 保存训练集预测结果
            train_pred_df = df_complete.iloc[self.train_indices].copy()
            train_missing_mask = missing_mask_new[self.train_indices]
            
            for j, col in enumerate(self.Y_columns):
                col_vals = train_pred_df[col].values.copy()
                col_vals[train_missing_mask[:, j]] = train_predictions[train_missing_mask[:, j], j]
                train_pred_df[col] = col_vals
            
            train_pred_path = os.path.join(save_dir, f"{base_name}_train_predictions.csv")
            train_pred_df.to_csv(train_pred_path, index=False, encoding='utf-8-sig')
            print(f"💾 已保存训练集预测: {train_pred_path}")

            # 8) 注意力权重
            if self.attention_weights is not None:
                attn_mean = np.mean(self.attention_weights, axis=0)
                attn_df = pd.DataFrame(attn_mean.reshape(1, -1), 
                                      columns=[f"a_{i}" for i in range(attn_mean.shape[0])])
                attn_path = os.path.join(save_dir, f"{base_name}_attention_mean.csv")
                attn_df.to_csv(attn_path, index=False, encoding='utf-8-sig')
                print(f"💾 已保存注意力均值: {attn_path}")

            # 9) 保存分类概率
            if hasattr(self, 'class_probs') and self.class_probs is not None:
                avg_class_probs = np.mean(self.class_probs, axis=0)
                class_prob_df = pd.DataFrame(
                    avg_class_probs,
                    columns=['P(min-Q1)', 'P(Q1-Q2)', 'P(Q2-Q3)', 'P(Q3-max)', 'P(outlier)'],
                    index=self.Y_columns
                )
                class_prob_path = os.path.join(save_dir, f"{base_name}_class_probabilities.csv")
                class_prob_df.to_csv(class_prob_path, encoding='utf-8-sig')
                print(f"💾 已保存分类概率: {class_prob_path}")

            print("\n" + "="*60)
            print("🎉 流程完成！")
            print("="*60)
            print(f"📁 输出目录: {save_dir}")
            print("\n✅ 关键输出文件:")
            print(f"   1. 训练集评估: train_set_evaluation_masked.csv")
            print(f"   2. 测试集评估: test_set_evaluation.csv")
            print(f"   3. 性能对比: train_test_comparison.csv")
            print(f"   4. 训练集预测: {base_name}_train_predictions.csv")
            print(f"   5. 测试集预测: {base_name}_test_predictions.csv")
            print(f"   6. 可视化图表: train_set_predictions_masked.png & test_set_predictions.png")
            print("\n⚠️  无泄漏保证:")
            print("   ✓ 训练集评估：仅在被mask位置（训练时未见真值）")
            print("   ✓ 测试集评估：完全未见过的样本")
            print("   ✓ 标准化器：仅在训练集上训练")
            print("   ✓ 四分位数边界：仅用训练集计算")
            print("   ✓ 中位数填充：仅用训练集统计")

        except Exception as e:
            print(f"\n❌运行错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None


# 在 Jupyter Notebook 中运行

# 1. 初始化预测器
"""
POLYBERT_MODEL_PATH = "/root/autodl-tmp/polyBERT"

predictor = SimpleSMILESYPredictor(
    random_state=42,
    alpha_non_missing_loss=0.3,
    lambda_vae=0.08,
    beta_vae=0.08,
    vae_latent_dim=64,
    bert_model_path=POLYBERT_MODEL_PATH,
    lambda_class=0.7,
    lambda_cascade=[1, 0.08, 0.6, 1.0],
    test_size=0.2  # 20%测试集
)

# 2. 配置数据路径
data_dict = {
    'ele':'Table_Electronic.csv',
    'energy':'Table_Energy.csv',
    'barrer':'Table_Permeability_Barrer.csv'
}
smiles_col_dict = {
    'ele': 'SMILES',
    'energy': 'SMILES',
    'barrer': 'SMILES'
}

# 3. 运行实验
task_key = 'ele'
csv_path = data_dict[task_key]
smiles_col = smiles_col_dict[task_key]

# 4. 测试不同保留策略
keep_modes = ['keep1', 'keep25', 'keep50', 'keep75']
for km in keep_modes:
    print(f"\n{'='*60}")
    print(f"🧪 运行实验: {task_key} - {km}")
    print(f"{'='*60}\n")
    
    predictor.run_pipeline(
        csv_file_path=csv_path,
        smiles_column=smiles_col,
        keep_mode=km,
        task_key=task_key
    )
"""

TASK_CONFIG = {
    'ele': {'csv': 'Table_Electronic.csv', 'smiles': 'SMILES'},
    'energy': {'csv': 'Table_Energy.csv', 'smiles': 'SMILES'},
    'barrer': {'csv': 'Table_Permeability_Barrer.csv', 'smiles': 'SMILES'},
    'dft': {'csv': 'DFT_properties_simple.csv', 'smiles': 'SMILES'},
    'md': {'csv': 'MD_properties_simple.csv', 'smiles': 'SMILES'},
    'newmd': {'csv': 'polymer_MD.csv', 'smiles': 'smiles_list'},
    'qc': {'csv': 'calculated_polymer_data.csv', 'smiles': 'psmiles'},
}
DEFAULT_KEEP_MODES = ['keep1', 'keep25', 'keep50', 'keep75']


def create_predictor():
    return SimpleSMILESYPredictor(
        random_state=42,
        alpha_non_missing_loss=0.3,
        lambda_vae=0.08,
        beta_vae=0.08,
        vae_latent_dim=64,
        bert_model_path=POLYBERT_MODEL_PATH,
        lambda_class=0.7,
        lambda_cascade=[1, 0.08, 0.6, 1.0],
        test_size=0.2
    )


def resolve_task_config(task_key):
    if task_key not in TASK_CONFIG:
        valid = ', '.join(sorted(TASK_CONFIG))
        raise ValueError(f"Unknown task_key '{task_key}'. Valid options: {valid}")

    task_cfg = TASK_CONFIG[task_key]
    return {
        'csv_path': os.path.join(DATA_DIR, task_cfg['csv']),
        'smiles_col': task_cfg['smiles'],
    }


def run_experiment(task_key='ele', keep_modes=None):
    config = resolve_task_config(task_key)
    predictor = create_predictor()
    keep_modes = keep_modes or DEFAULT_KEEP_MODES

    print(f"Using data directory: {DATA_DIR}")
    print(f"Using model directory: {POLYBERT_MODEL_PATH}")
    print(f"Using ckpt directory: {CKPT_DIR}")

    for km in keep_modes:
        print(f"\n{'=' * 60}")
        print(f"Running experiment: {task_key} - {km}")
        print(f"{'=' * 60}\n")

        predictor.run_pipeline(
            csv_file_path=config['csv_path'],
            smiles_column=config['smiles_col'],
            keep_mode=km,
            task_key=task_key
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Unified polyBERT cascade runner')
    parser.add_argument(
        '--task',
        choices=sorted(TASK_CONFIG),
        default='ele',
        help='Task key to run'
    )
    parser.add_argument(
        '--keep-modes',
        nargs='+',
        default=DEFAULT_KEEP_MODES,
        choices=DEFAULT_KEEP_MODES,
        help='Keep strategies to evaluate'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(task_key=args.task, keep_modes=args.keep_modes)


if __name__ == '__main__':
    main()
