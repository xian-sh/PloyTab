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
        if 'Training Loss Curve' in label:
            label = 'Training Loss Curve (Train Set Only)'
    return _original_plt_title(label, *args, **kwargs)

plt.title = _safe_plt_title

# BERT-related libraries
try:
    from transformers import AutoTokenizer, AutoModel
    print("✅ Transformers imported")
except ImportError:
    print("❌ transformers is required: pip install transformers")
    raise

def _basename_no_ext(path):
    base = os.path.basename(path)
    if '.' in base:
        base = '.'.join(base.split('.')[:-1])
    return base

class BERTFeatureExtractor:
    """polyBERT feature extractor"""
    def __init__(self, model_path, max_length=512):
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load the polyBERT model"""
        try:
            print(f"🤗 Loading polyBERT model: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("✅ polyBERT model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load polyBERT model: {e}")
            raise
    
    def extract_cls_token(self, smiles):
        """Extract CLS-token features for one SMILES string"""
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
            print(f"⚠️ polyBERT feature extraction failed: {e}")
            hidden_size = self.model.config.hidden_size
            return np.zeros(hidden_size, dtype=np.float32)
    
    def get_features(self, smiles_list):
        """Batch extract polyBERT features"""
        print("🧬 Generating polyBERT molecular features...")
        features = []
        valid_count = 0
        
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(smiles_list)}")
            
            feature = self.extract_cls_token(smiles)
            features.append(feature)
            if not np.all(feature == 0):
                valid_count += 1
        
        features = np.array(features, dtype=np.float32)
        print(f"✅ polyBERT feature shape: {features.shape}")
        print(f"✅ Valid molecule count: {valid_count}/{len(features)}")
        return features

class MolecularDataset(Dataset):
    """Molecular dataset class with classification labels"""
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
    """Simple attention module for Y-variable relationships"""
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

class PriorFeatureCNN(nn.Module):
    """Construct tabular prior features with a 1 x 3 x 3 convolution.

    The tensor is arranged as [batch, channel=1, depth=1, rows=3, properties],
    where the three rows correspond to observed/input values, the previous estimate,
    and the missingness mask. The convolution is applied over this local
    tabular-context representation, not over polymer sample order.
    """
    def __init__(self, y_dim, output_dim, conv_channels=16, dropout=0.1):
        super().__init__()
        self.y_dim = y_dim
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=(1, 1, 3),
                padding=(0, 0, 1),
            ),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels * 3 * y_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, observed_values, previous_estimate, missing_mask):
        mask_values = missing_mask.float()
        h = torch.stack([observed_values, previous_estimate, mask_values], dim=1)
        h = h.unsqueeze(1).unsqueeze(2)
        return self.proj(self.conv(h))

class CascadeRegressionHead(nn.Module):
    """Cascade regression head with CNN prior-feature construction."""
    def __init__(self, input_dim, y_dim, hidden_dim=128):
        super().__init__()
        self.y_dim = y_dim
        self.prior_cnn = PriorFeatureCNN(y_dim=y_dim, output_dim=hidden_dim)

        head_input_dim = input_dim + hidden_dim + y_dim
        self.reg_head_1 = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, y_dim),
        )
        self.reg_head_2 = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, y_dim),
        )
        self.reg_head_3 = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, y_dim),
        )
        self.reg_head_4 = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, y_dim),
        )

    def _refine_once(self, head, fused_features, observed_values, previous_estimate, missing_mask):
        prior_features = self.prior_cnn(observed_values, previous_estimate, missing_mask)
        reg_input = torch.cat([fused_features, prior_features, previous_estimate], dim=1)
        residual = head(reg_input)
        return previous_estimate + residual

    def forward(self, fused_features, class_init, observed_values, missing_mask):
        """Refine interval initialization with CNN prior features and residual heads."""
        pred_1 = self._refine_once(self.reg_head_1, fused_features, observed_values, class_init, missing_mask)
        pred_2 = self._refine_once(self.reg_head_2, fused_features, observed_values, pred_1, missing_mask)
        pred_3 = self._refine_once(self.reg_head_3, fused_features, observed_values, pred_2, missing_mask)
        pred_4 = self._refine_once(self.reg_head_4, fused_features, observed_values, pred_3, missing_mask)
        return pred_1, pred_2, pred_3, pred_4

class SimpleSMILESPredictor(nn.Module):
    """SMILES-Y prediction network with classification and cascade regression heads"""
    def __init__(self, feature_dim, y_dim, hidden_dim=256, dropout=0.2, num_classes=4,
                 interval_centers=None):
        super(SimpleSMILESPredictor, self).__init__()
        self.feature_dim = feature_dim
        self.y_dim = y_dim
        self.num_classes = num_classes
        if interval_centers is None:
            interval_centers = torch.zeros(y_dim, num_classes)
        self.register_buffer('interval_centers', torch.as_tensor(interval_centers, dtype=torch.float32))
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Property-context attention over values and missingness indicators.
        self.y_attention = SimpleYAttention(y_dim * 2)
        
        # Y encoder
        self.y_encoder = nn.Sequential(
            nn.Linear(y_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2.0)
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head: predict the interval for each Y variable
        self.classification_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            ) for _ in range(y_dim)
        ])
        
        # Cascade regression head
        self.cascade_regression = CascadeRegressionHead(
            input_dim=hidden_dim * 2,
            y_dim=y_dim,
            hidden_dim=hidden_dim
        )
        
        print(f"📊 Model parameters: feature_dim={feature_dim}, y_dim={y_dim}, hidden_dim={hidden_dim}")
        print(f"🔢 Classification head: {num_classes} intervals")
        print(f"🎯 Regression head: 4-stage cascade refinement")
        
    def forward(self, features, y_features, missing_mask):
        # Feature encoding
        feature_encoded = self.feature_encoder(features)
        
        # Y attention enhancement
        mask_features = missing_mask.float()
        y_context = torch.cat([y_features, mask_features], dim=1)
        y_enhanced, attention_weights = self.y_attention(y_context)
        y_encoded = self.y_encoder(y_enhanced)
        
        # Feature fusion
        fused = torch.cat([feature_encoded, y_encoded], dim=1)
        fused_features = self.fusion(fused)
        
        # Classification head
        class_logits = []
        for i in range(self.y_dim):
            logits = self.classification_head[i](fused_features)
            class_logits.append(logits)
        class_logits = torch.stack(class_logits, dim=1)
        class_probs = torch.softmax(class_logits, dim=-1)
        class_init = torch.sum(class_probs * self.interval_centers.unsqueeze(0), dim=-1)
        
        # Cascade regression head
        pred_1, pred_2, pred_3, pred_4 = self.cascade_regression(
            fused_features, class_init, y_features, missing_mask
        )
        
        return {
            'class_logits': class_logits,
            'class_probs': class_probs,
            'class_init': class_init,
            'predictions_1': pred_1,
            'predictions_2': pred_2,
            'predictions_3': pred_3,
            'predictions_4': pred_4,
            'attention_weights': attention_weights
        }

class SimpleSMILESYPredictor:
    def __init__(self, device=None, random_state=42, alpha_non_missing_loss=0.2,
                 bert_model_path=None,
                 lambda_class=1.0, lambda_cascade=[0.8, 0.6, 0.4, 1.0],
                 test_size=0.2):
        """
        Parameters:
        - test_size: test-set ratio (default 0.2, i.e., 20%)
        - Other parameters remain unchanged
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        if bert_model_path is None:
            raise ValueError("polyBERT model path must be provided (bert_model_path)")
        
        self.feature_generator = BERTFeatureExtractor(bert_model_path)
        self.feature_dim = None
        
        self.scaler_Y_input = StandardScaler()
        self.scaler_Y_target = StandardScaler()
        
        self.model = None
        self.Y_columns = []
        self.low_variance_cols = []
        self.alpha_non_missing_loss = alpha_non_missing_loss
        self.attention_weights = None
        
        self.lambda_class = lambda_class
        self.lambda_cascade = lambda_cascade
        
        self.quantile_boundaries = None
        
        # Added: train/test indices
        self.test_size = test_size
        self.train_indices = None
        self.test_indices = None
        self._train_medians = None  # Store training-set medians
        
        print(f"🖥️ Using device: {self.device}")
        print(f"🎯 Loss weights: lambda_class={lambda_class}, lambda_cascade={lambda_cascade}")
        print(f"🔀 Dataset split: train={1-test_size:.0%}, test={test_size:.0%}")
        print(f"🧬 Using polyBERT feature extraction")
    
    def _build_keep_mask(self, n_rows, n_cols, keep_k, rng):
        """Keep keep_k columns for each row and mark the rest as missing"""
        keep_k = int(max(1, min(keep_k, n_cols - 1)))
        missing_mask_new = np.ones((n_rows, n_cols), dtype=bool)
        for i in range(n_rows):
            keep_cols = rng.choice(n_cols, size=keep_k, replace=False)
            missing_mask_new[i, keep_cols] = False
        return missing_mask_new, keep_k

    def _compute_quantile_boundaries(self, Y_data, missing_mask_new):
        """Compute quartile boundaries for each column using retained values only"""
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
        
        print("\n📊 Quartile boundaries computed from retained training values only:")
        for col, bounds in quantile_boundaries.items():
            print(f"  {col}: min={bounds[0]:.3f}, Q1={bounds[1]:.3f}, Q2={bounds[2]:.3f}, "
                  f"Q3={bounds[3]:.3f}, max={bounds[4]:.3f}")
        
        return quantile_boundaries
    
    def _assign_class_labels(self, Y_data, missing_mask_new):
        """Assign each property value to one of four quartile intervals (0-3)."""
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

    def _compute_interval_centers_scaled(self):
        """Return scaled representative values for the four quartile intervals."""
        centers = []
        for j, col in enumerate(self.Y_columns):
            q_min, q25, q50, q75, q_max = self.quantile_boundaries[col]
            raw_centers = np.array([
                0.5 * (q_min + q25),
                0.5 * (q25 + q50),
                0.5 * (q50 + q75),
                0.5 * (q75 + q_max),
            ], dtype=np.float32)
            scale = self.scaler_Y_target.scale_[j]
            if scale == 0:
                scale = 1.0
            centers.append((raw_centers - self.scaler_Y_target.mean_[j]) / scale)
        return np.vstack(centers).astype(np.float32)

    def load_and_process_data(self, csv_file_path, smiles_column='psmiles',
                              keep_mode='keep1', save_files=True, task_key=None):
        """Load and process data with strict train/test separation"""
        print(f"📂 Loading data: {csv_file_path}")
        print("🔒 Strict mode: train and test sets are fully separated, with no data leakage")

        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        df_raw = None
        for encoding in encodings:
            try:
                df_raw = pd.read_csv(csv_file_path, encoding=encoding)
                print(f"✅ Read successfully (encoding: {encoding})")
                break
            except UnicodeDecodeError:
                continue
        if df_raw is None:
            raise ValueError("Unable to read CSV file")

        unnamed_cols = [c for c in df_raw.columns if str(c).startswith('Unnamed:')]
        if unnamed_cols:
            print(f"🧹 Removed unnamed columns: {unnamed_cols}")
            df_raw = df_raw.drop(columns=unnamed_cols)

        print(f"📊 Raw data: {df_raw.shape}")
        if smiles_column not in df_raw.columns:
            raise ValueError(f"SMILES column not found: {smiles_column}")

        # Identify candidate Y columns
        candidate_cols = []
        for col in df_raw.columns:
            if col == smiles_column:
                continue
            s = pd.to_numeric(df_raw[col], errors='coerce')
            if not s.isnull().all():
                candidate_cols.append(col)

        # Drop rows containing NaN values
        y_df_numeric = df_raw[candidate_cols].apply(pd.to_numeric, errors='coerce')
        non_na_mask = ~y_df_numeric.isnull().any(axis=1)
        df = df_raw.loc[non_na_mask].reset_index(drop=True)
        y_df_numeric = df[candidate_cols].apply(pd.to_numeric, errors='coerce')
        print(f"🧹 Data after dropping samples with NaN values: {df.shape}")

        # Statistics and low-variance filtering
        stats = []
        for col in candidate_cols:
            vals = pd.to_numeric(df[col], errors='coerce').values
            std = float(np.std(vals)) if len(vals) > 0 else 0.0
            uniq = int(len(np.unique(vals))) if len(vals) > 0 else 0
            stats.append((col, std, uniq))
        self.low_variance_cols = [c for c, std, uniq in stats if std < 1e-8 or uniq <= 1]
        self.Y_columns = [c for c, std, uniq in stats if c not in self.low_variance_cols]
        
        print("🔬 Column statistics:")
        for c, std, uniq in stats:
            tag = "LOW-VAR" if c in self.low_variance_cols else ""
            print(f"  - {c}: std={std:.3e}, nunique={uniq} {tag}")
        print(f"✅ Number of valid Y columns: {len(self.Y_columns)}")
        
        if len(self.Y_columns) == 0:
            raise ValueError("No valid Y columns are available")

        # === Key change 1: split train/test sets first ===
        n_samples = len(df)
        all_indices = np.arange(n_samples)
        
        self.train_indices, self.test_indices = train_test_split(
            all_indices,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )
        
        print(f"\n🔀 Dataset split:")
        print(f"   Train set: {len(self.train_indices)} samples ({(1-self.test_size)*100:.0f}%)")
        print(f"   Test set: {len(self.test_indices)} samples ({self.test_size*100:.0f}%)")
        print("   ⚠️  The test set is fully excluded from training and standardization\n")

        # polyBERT feature extraction
        smiles_list = df[smiles_column].tolist()
        features = self.feature_generator.get_features(smiles_list)
        
        if len(features) > 0:
            self.feature_dim = features.shape[1]
            print(f"✅ Dynamically detected polyBERT feature dimension: {self.feature_dim}")

        y_data = df[self.Y_columns].values.astype(np.float32)
        n_rows, n_cols = y_data.shape

        # Determine the retention ratio
        if keep_mode == 'keep1':
            keep_k = 1
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
        print(f"📌 Retention strategy: {keep_mode} -> retain per row {keep_k}/{n_cols} values")

        # === Key change 2: compute quartile boundaries using training data only ===
        y_data_train = y_data[self.train_indices]
        missing_mask_train = missing_mask_new[self.train_indices]
        
        self._compute_quantile_boundaries(y_data_train, missing_mask_train)
        
        # Assign classification labels for all data using boundaries from the training set only
        class_labels = self._assign_class_labels(y_data, missing_mask_new)
        print(f"✅ Classification labels generated: {class_labels.shape}")

        # Construct training-input version with NaN at non-retained positions
        df_missing_for_training = df.copy()
        y_data_missing = y_data.copy().astype(object)
        y_data_missing[missing_mask_new] = np.nan
        for j, col in enumerate(self.Y_columns):
            df_missing_for_training[col] = y_data_missing[:, j]

        # Save directory
        base_name = _basename_no_ext(csv_file_path)
        task_key = task_key if task_key else base_name
        save_dir = os.path.join(CKPT_DIR, f"{task_key}_polyBERT_cascade_{keep_mode}_260107_SAFE")
        os.makedirs(save_dir, exist_ok=True)
        print(f"🗂️ Save directory: {save_dir}")

        # Save train/test ground-truth files separately
        df_train = df.iloc[self.train_indices].reset_index(drop=True)
        df_test = df.iloc[self.test_indices].reset_index(drop=True)
        
        train_gt_path = os.path.join(save_dir, f"{base_name}_train_GT.csv")
        test_gt_path = os.path.join(save_dir, f"{base_name}_test_GT.csv")
        
        df_train.to_csv(train_gt_path, index=False, encoding='utf-8-sig')
        df_test.to_csv(test_gt_path, index=False, encoding='utf-8-sig')
        
        print(f"💾 Training-set ground truth saved: {train_gt_path}")
        print(f"💾 Test-set ground truth saved: {test_gt_path}")

        # Save the version with missing values
        df_missing_to_save = df.copy().astype(object)
        for j, col in enumerate(self.Y_columns):
            # col_vals = df_missing_to_save[col].values
            col_vals = df_missing_to_save[col].to_numpy(dtype=object, copy=True)
            write_empty = missing_mask_new[:, j]
            col_vals[write_empty] = ""
            df_missing_to_save[col] = col_vals

        missing_path = os.path.join(save_dir, f"{base_name}_{keep_mode}.csv")
        df_missing_to_save.to_csv(missing_path, index=False, encoding='utf-8-sig')
        print(f"💾 Retention-strategy CSV saved: {missing_path}")

        total_missing = missing_mask_new.sum()
        overall_rate = total_missing / (n_rows * n_cols) * 100.0
        print(f"📊 Masked ratio: {overall_rate:.2f}%")

        original_nan_mask = np.zeros_like(missing_mask_new, dtype=bool)

        return df, df_missing_for_training, features, missing_mask_new, original_nan_mask, save_dir, class_labels
    
    def train_model(self, features, df_complete, df_missing, missing_mask_new, class_labels,
                    epochs=200, batch_size=32, lr=0.001):
        """Train the model using only the training set, without data leakage"""
        print("\n🚀 Starting model training using only the training set; the test set is fully isolated...")
        print("⚠️  Key point: scalers, quartiles, and other statistics are computed only on the training set")
        
        # === Key change: use training-set data only ===
        train_features = features[self.train_indices]
        train_Y_complete = df_complete.iloc[self.train_indices][self.Y_columns].values.astype(np.float32)
        train_Y_missing = df_missing.iloc[self.train_indices][self.Y_columns].values.astype(np.float32)
        train_missing_mask = missing_mask_new[self.train_indices]
        train_class_labels = class_labels[self.train_indices]
        
        print(f"📊 Training-set size: {len(self.train_indices)} samples")
        print(f"🔒 Test-set size: {len(self.test_indices)} samples(completely unused)")
        
        missing_mask = np.isnan(train_Y_missing) | np.isnan(train_Y_complete)

        # Fill missing values using medians computed only from retained training data
        Y_data_filled = train_Y_missing.copy()
        self._train_medians = []  # Save training-set medians for test-set use
        
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
        
        print(f"✅ Training-set medians saved: {self._train_medians[:3]}...")
        
        # === Key change: fit scalers only on retained training data ===
        print("📊 Standardizing data using retained training values only...")
        
        # Input-side standardization
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
        
        # Target-side standardization
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
        interval_centers_scaled = self._compute_interval_centers_scaled()
        
        print(f"✅ Input scaler statistics (training set only): mean={self.scaler_Y_input.mean_[:3]}, scale={self.scaler_Y_input.scale_[:3]}")
        print(f"✅ Target scaler statistics (training set only): mean={self.scaler_Y_target.mean_[:3]}, scale={self.scaler_Y_target.scale_[:3]}")
        
        dataset = MolecularDataset(train_features, Y_scaled_target, Y_scaled_input, missing_mask, train_class_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Build model
        self.model = SimpleSMILESPredictor(
            feature_dim=self.feature_dim,
            y_dim=len(self.Y_columns),
            hidden_dim=256,
            dropout=0.2,
            num_classes=4,
            interval_centers=interval_centers_scaled
        ).to(self.device)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=15, min_lr=1e-6
        )
        
        self.model.train()
        losses = []
        best_loss = float('inf')
        patience = 0
        
        print(f"🏋️ Training parameters: epochs={epochs}, batch_size={batch_size}, lr={lr}")
        print(f"🎯 Loss weights: lambda_class={self.lambda_class}, lambda_cascade={self.lambda_cascade}")
        
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
                
                # Classification loss
                class_logits = outputs['class_logits']
                class_loss = 0.0
                for i in range(len(self.Y_columns)):
                    class_loss += nn.CrossEntropyLoss()(
                        class_logits[:, i, :],
                        class_labels_batch[:, i]
                    )
                class_loss = class_loss / len(self.Y_columns)
                
                # Cascade regression loss
                reg_losses = []
                for layer_idx, pred_key in enumerate(['predictions_1', 'predictions_2', 
                                                      'predictions_3', 'predictions_4']):
                    pred = outputs[pred_key]
                    reg_loss = self.compute_loss_weighted(
                        pred, y_target, missing_mask_batch, alpha=self.alpha_non_missing_loss
                    )
                    reg_losses.append(reg_loss * self.lambda_cascade[layer_idx])
                
                # Total loss
                loss = (self.lambda_class * class_loss + 
                        sum(reg_losses))
                
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
                    print(f"   Total loss: {avg_loss:.6f}")
                    print(f"   Classification loss: {avg_class_loss:.6f}")
                    print(f"   Regression loss: L1={avg_reg_losses[0]:.6f}, L2={avg_reg_losses[1]:.6f}, "
                          f"L3={avg_reg_losses[2]:.6f}, L4={avg_reg_losses[3]:.6f}")
                    print(f"   LR: {optimizer.param_groups[0]['lr']:.8f}")
                
                if patience >= 30:
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    break
        
        print("✅ Training complete!")
        return losses
    
    def compute_loss_weighted(self, predictions, targets, missing_mask, alpha=0.2):
        """Weighted MSE"""
        w = torch.where(missing_mask, torch.ones_like(targets), torch.full_like(targets, alpha))
        mse = (w * (predictions - targets) ** 2).sum() / w.sum().clamp_min(1.0)
        return mse

    def evaluate_on_train_set(self, features, df_complete, df_missing, missing_mask_new):
        """Evaluate on the training set only at masked positions, without leakage"""
        print("\n📊 Evaluating on the training set at positions masked during training...")
        print("⚠️  Evaluation positions: missing positions unseen during training, with no leakage")
    
        if self.model is None:
            raise ValueError("The model must be trained first")
    
        # === Use training-set data only ===
        train_features = features[self.train_indices]
        train_Y_complete = df_complete.iloc[self.train_indices][self.Y_columns].values.astype(np.float32)
        train_Y_missing = df_missing.iloc[self.train_indices][self.Y_columns].values.astype(np.float32)
        train_missing_mask = missing_mask_new[self.train_indices]
    
        print(f"🔬 Number of training samples: {len(self.train_indices)}")
        print(f"📌 Evaluation strategy: compute errors only at masked positions whose ground truth was unseen during training")
    
        # Fill missing values in the training set
        Y_data_filled = train_Y_missing.copy()
        for i in range(Y_data_filled.shape[1]):
            fill_value = self._train_medians[i]
            Y_data_filled[np.isnan(Y_data_filled[:, i]), i] = fill_value
    
        # Standardization
        Y_scaled_input = self.scaler_Y_input.transform(Y_data_filled)
    
        # Prediction
        self.model.eval()
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(train_features).to(self.device)
            y_tensor = torch.FloatTensor(Y_scaled_input).to(self.device)
            dummy_mask = torch.zeros_like(y_tensor, dtype=torch.bool).to(self.device)
        
            outputs = self.model(feature_tensor, y_tensor, dummy_mask)
            predictions = outputs['predictions_4']
            main_pred = self.scaler_Y_target.inverse_transform(predictions.cpu().numpy())
    
        fused = main_pred
    
        # Key: evaluate only at masked positions whose ground truth was unseen during training
        results = {}
        for i, col in enumerate(self.Y_columns):
            mask = train_missing_mask[:, i]  # Positions masked during training
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
    
        print("\n🎯 Training-set evaluation results at masked positions only:")
        if results:
            results_df = pd.DataFrame(results).T
            print(results_df[['R²', 'Correlation', 'MAE', 'N_train_masked']].round(4))
            print(f"\nMean R²: {results_df['R²'].mean():.4f}")
            print(f"Mean correlation: {results_df['Correlation'].mean():.4f}")
            print(f"Mean relative error: {results_df['Relative_Error_%'].mean():.2f}%")
    
        return results, fused

    def visualize_train_results(self, train_predictions, df_complete, missing_mask_new, save_dir="."):
        """Visualize training-set results at masked positions only"""
        print("📊 Generating training-set visualization at masked positions only...")
    
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
        print(f"💾 Training-set plot saved: {fig_path}")
    
    def evaluate_on_test_set(self, features, df_complete, df_missing, missing_mask_new):
        """Evaluate on the completely unseen test set"""
        print("\n📊 Evaluating on the completely unseen test set...")
        
        if self.model is None:
            raise ValueError("The model must be trained first")
        
        # === Use test-set data only ===
        test_features = features[self.test_indices]
        test_Y_complete = df_complete.iloc[self.test_indices][self.Y_columns].values.astype(np.float32)
        test_Y_missing = df_missing.iloc[self.test_indices][self.Y_columns].values.astype(np.float32)
        test_missing_mask = missing_mask_new[self.test_indices]
        
        print(f"🔬 Number of test samples: {len(self.test_indices)}")
        
        # Fill missing values in the test set using medians learned from the training set
        Y_data_filled = test_Y_missing.copy()
        for i in range(Y_data_filled.shape[1]):
            col_values = Y_data_filled[:, i]
            fill_value = self._train_medians[i]  # Use training-set median
            Y_data_filled[np.isnan(Y_data_filled[:, i]), i] = fill_value
        
        # Use trained scalers already fitted on the training set
        Y_scaled_input = self.scaler_Y_input.transform(Y_data_filled)
        
        # Prediction
        self.model.eval()
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(test_features).to(self.device)
            y_tensor = torch.FloatTensor(Y_scaled_input).to(self.device)
            dummy_mask = torch.zeros_like(y_tensor, dtype=torch.bool).to(self.device)
            
            outputs = self.model(feature_tensor, y_tensor, dummy_mask)
            predictions = outputs['predictions_4']
            main_pred = self.scaler_Y_target.inverse_transform(predictions.cpu().numpy())
            
            # Save attention weights and classification probabilities
            self.attention_weights = outputs['attention_weights'].cpu().numpy()
            self.class_probs = outputs['class_probs'].cpu().numpy()
        
        fused = main_pred
        
        # Compute metrics only at masked positions
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
        
        print("\n🎯 Test-set evaluation results:")
        if results:
            results_df = pd.DataFrame(results).T
            print(results_df[['R²', 'Correlation', 'MAE', 'N_test_missing']].round(4))
            print(f"\nMean R²: {results_df['R²'].mean():.4f}")
            print(f"Mean correlation: {results_df['Correlation'].mean():.4f}")
            print(f"Mean relative error: {results_df['Relative_Error_%'].mean():.2f}%")
        
        return results, fused
    
    def visualize_test_results(self, test_predictions, df_complete, missing_mask_new, save_dir="."):
        """Visualize test-set results"""
        print("📊 Generating test-set visualization...")
        
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
        print(f"💾 Test-set plot saved: {fig_path}")

    def plot_training_curve(self, losses, save_dir="."):
        if losses:
            plt.figure(figsize=(10, 6))
            plt.plot(losses, 'b-', linewidth=2, label='Total Loss')
            plt.title('Training Loss Curve (training set only)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            fig_path = os.path.join(save_dir, 'training_curve.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"💾 Curve saved: {fig_path}")

    def run_pipeline(self, csv_file_path, smiles_column='psmiles',
                     keep_mode='keep1', task_key=None):
        """Full pipeline with strict train/test separation"""
        print("=" * 60)
        print("🧪 polyBERT-Cascade prediction system (no-leakage version)")
        print("🔬 Strict train/test separation + leakage-free standardization")
        print("=" * 60)
        
        try:
            # 1) Data loading with train/test split
            (df_complete, df_missing_for_training, features,
             missing_mask_new, original_nan_mask, save_dir, class_labels) = self.load_and_process_data(
                csv_file_path=csv_file_path,
                smiles_column=smiles_column,
                keep_mode=keep_mode,
                save_files=True,
                task_key=task_key
            )
            base_name = _basename_no_ext(csv_file_path)

            # 2) Train the model using only the training set
            losses = self.train_model(
                features, df_complete, df_missing_for_training, 
                missing_mask_new, class_labels, epochs=300, batch_size=16
            )

            # 3a) Evaluate on the training set only at masked positions, without leakage
            print("\n" + "="*60)
            print("📊 Evaluation stage 1: training-set performance at masked positions only")
            print("="*60)
            train_results, train_predictions = self.evaluate_on_train_set(
                features, df_complete, df_missing_for_training, missing_mask_new
            )
            
            # Visualize training-set results
            self.visualize_train_results(train_predictions, df_complete, missing_mask_new, save_dir=save_dir)
            
            # Save training-set evaluation results
            if train_results:
                train_results_df = pd.DataFrame(train_results).T
                train_eval_path = os.path.join(save_dir, 'train_set_evaluation_masked.csv')
                train_results_df.to_csv(train_eval_path, encoding='utf-8-sig')
                print(f"💾 Training-set evaluation results saved: {train_eval_path}")

            # 3b) Evaluate on the test set
            print("\n" + "="*60)
            print("📊 Evaluation stage 2: performance on completely unseen test data")
            print("="*60)
            test_results, test_predictions = self.evaluate_on_test_set(
                features, df_complete, df_missing_for_training, missing_mask_new
            )

            # 4) Visualize test-set results
            self.visualize_test_results(test_predictions, df_complete, missing_mask_new, save_dir=save_dir)
            self.plot_training_curve(losses, save_dir=save_dir)

            # 5) Compare train/test performance
            print("\n" + "="*60)
            print("📊 Performance comparison summary (Train vs Test)")
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
                
                print("\n📊 Detailed comparison:")
                print(comparison.round(4))
                
                print(f"\n🎯 Overall statistics:")
                print(f"   Mean train R²: {comparison['Train_R²'].mean():.4f}")
                print(f"   Mean test R²: {comparison['Test_R²'].mean():.4f}")
                print(f"   R² gap (Train-Test): {comparison['R²_Gap'].mean():.4f}")
                print(f"\n   Mean train correlation: {comparison['Train_Corr'].mean():.4f}")
                print(f"   Mean test correlation: {comparison['Test_Corr'].mean():.4f}")
                print(f"   Correlation gap: {comparison['Corr_Gap'].mean():.4f}")
                print(f"\n   Mean train MAE: {comparison['Train_MAE'].mean():.4f}")
                print(f"   Mean test MAE: {comparison['Test_MAE'].mean():.4f}")
                
                # Estimate overfitting level
                avg_gap = comparison['R²_Gap'].mean()
                if avg_gap < 0.05:
                    print(f"\n✅ Overfitting level: very low (R² gap={avg_gap:.4f})")
                elif avg_gap < 0.10:
                    print(f"\n⚠️  Overfitting level: mild (R² gap={avg_gap:.4f})")
                else:
                    print(f"\n❌ Overfitting level: high (R² gap={avg_gap:.4f})")
                
                # Save comparison results
                comp_path = os.path.join(save_dir, 'train_test_comparison.csv')
                comparison.to_csv(comp_path, encoding='utf-8-sig')
                print(f"\n💾 Comparison results saved: {comp_path}")

            # 6) Save test-set evaluation results
            if test_results:
                test_results_df = pd.DataFrame(test_results).T
                test_eval_path = os.path.join(save_dir, 'test_set_evaluation.csv')
                test_results_df.to_csv(test_eval_path, encoding='utf-8-sig')
                print(f"💾 Test-set evaluation results saved: {test_eval_path}")
                
                print("\n📊 Test-set performance summary:")
                print(test_results_df.round(4))
                print(f"\n🎯 Overall test-set performance:")
                print(f"   Mean R²: {test_results_df['R²'].mean():.3f}")
                print(f"   Mean correlation: {test_results_df['Correlation'].mean():.3f}")
                print(f"   Mean relative error: {test_results_df['Relative_Error_%'].mean():.1f}%")

            # 7) Save test-set predictions
            test_pred_df = df_complete.iloc[self.test_indices].copy()
            test_missing_mask = missing_mask_new[self.test_indices]
            
            for j, col in enumerate(self.Y_columns):
                col_vals = test_pred_df[col].values.copy()
                col_vals[test_missing_mask[:, j]] = test_predictions[test_missing_mask[:, j], j]
                test_pred_df[col] = col_vals
            
            test_pred_path = os.path.join(save_dir, f"{base_name}_test_predictions.csv")
            test_pred_df.to_csv(test_pred_path, index=False, encoding='utf-8-sig')
            print(f"💾 Test-set predictions saved: {test_pred_path}")
            
            # 7b) Save training-set predictions
            train_pred_df = df_complete.iloc[self.train_indices].copy()
            train_missing_mask = missing_mask_new[self.train_indices]
            
            for j, col in enumerate(self.Y_columns):
                col_vals = train_pred_df[col].values.copy()
                col_vals[train_missing_mask[:, j]] = train_predictions[train_missing_mask[:, j], j]
                train_pred_df[col] = col_vals
            
            train_pred_path = os.path.join(save_dir, f"{base_name}_train_predictions.csv")
            train_pred_df.to_csv(train_pred_path, index=False, encoding='utf-8-sig')
            print(f"💾 Training-set predictions saved: {train_pred_path}")

            # 8) Attention weights
            if self.attention_weights is not None:
                attn_mean = np.mean(self.attention_weights, axis=0)
                attn_df = pd.DataFrame(attn_mean.reshape(1, -1), 
                                      columns=[f"a_{i}" for i in range(attn_mean.shape[0])])
                attn_path = os.path.join(save_dir, f"{base_name}_attention_mean.csv")
                attn_df.to_csv(attn_path, index=False, encoding='utf-8-sig')
                print(f"💾 Mean attention weights saved: {attn_path}")

            # 9) Save classification probabilities
            if hasattr(self, 'class_probs') and self.class_probs is not None:
                avg_class_probs = np.mean(self.class_probs, axis=0)
                class_prob_df = pd.DataFrame(
                    avg_class_probs,
                    columns=['P(min-Q1)', 'P(Q1-Q2)', 'P(Q2-Q3)', 'P(Q3-max)'],
                    index=self.Y_columns
                )
                class_prob_path = os.path.join(save_dir, f"{base_name}_class_probabilities.csv")
                class_prob_df.to_csv(class_prob_path, encoding='utf-8-sig')
                print(f"💾 Classification probabilities saved: {class_prob_path}")

            print("\n" + "="*60)
            print("🎉 Pipeline complete!")
            print("="*60)
            print(f"📁 Output directory: {save_dir}")
            print("\n✅ Key output files:")
            print(f"   1. Training-set evaluation: train_set_evaluation_masked.csv")
            print(f"   2. Test-set evaluation: test_set_evaluation.csv")
            print(f"   3. Performance comparison: train_test_comparison.csv")
            print(f"   4. Training-set predictions: {base_name}_train_predictions.csv")
            print(f"   5. Test-set predictions: {base_name}_test_predictions.csv")
            print(f"   6. Visualization plots: train_set_predictions_masked.png & test_set_predictions.png")
            print("\n⚠️  No-leakage guarantees:")
            print("   ✓ Training-set evaluation：only at masked positions whose ground truth was unseen during training")
            print("   ✓ Test-set evaluation：completely unseen samples")
            print("   ✓ scalers: fitted only on the training set")
            print("   ✓ quartile boundaries: computed only from the training set")
            print("   ✓ median imputation: computed only from the training set")

            return test_results, test_predictions

        except Exception as e:
            print(f"\n❌ Runtime error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None


# Run in Jupyter Notebook

# 1. Initialize predictor
"""
POLYBERT_MODEL_PATH = "/root/autodl-tmp/polyBERT"

predictor = SimpleSMILESYPredictor(
    random_state=42,
    alpha_non_missing_loss=0.3,
    bert_model_path=POLYBERT_MODEL_PATH,
    lambda_class=0.7,
    lambda_cascade=[1, 0.08, 0.6, 1.0],
    test_size=0.2  # 20% test set
)

# 2. Configure data paths
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

# 3. Run experiment
task_key = 'ele'
csv_path = data_dict[task_key]
smiles_col = smiles_col_dict[task_key]

# 4. Test different retention strategies
keep_modes = ['keep1', 'keep25', 'keep50', 'keep75']
for km in keep_modes:
    print(f"\n{'='*60}")
    print(f"🧪 Running experiment: {task_key} - {km}")
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
