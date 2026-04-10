"""
对比算法实现:
  传统方法:   PCA+Mahalanobis, PLS-DA, SVM-RBF, RandomForest
  深度学习:   ResNet-CE, ResNet-Triplet, ResNet-CenterLoss
  (消融变体在 compare.py 中通过配置控制)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader

from models import ResBlock2D, _make_stage


# ═══════════════════════════════════════════════════════════
#  特征提取
# ═══════════════════════════════════════════════════════════

def extract_features(loader):
    """从 DataLoader 提取扁平化特征、标签和批次。"""
    all_x, all_y, all_b = [], [], []
    for batch in loader:
        x = batch["input"].numpy()
        B = x.shape[0]
        all_x.append(x.reshape(B, -1))
        all_y.append(batch["product"].numpy())
        all_b.append(batch["batch"].numpy())
    return np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_b)


# ═══════════════════════════════════════════════════════════
#  传统方法基类
# ═══════════════════════════════════════════════════════════

class TraditionalBaseline:
    """传统方法统一接口。"""
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        """返回 (preds, scores)。"""
        raise NotImplementedError

    def get_embeddings(self, X):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────
#  PCA + Mahalanobis 距离
# ─────────────────────────────────────────────────────────

class PCAMahalanobis(TraditionalBaseline):
    """PCA 降维 → Mahalanobis 距离到各类质心 → 最近类为预测。"""

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = None
        self.class_stats = {}   # cls -> (mean, cov_inv)
        self._med_dist = 1.0

    def fit(self, X, y):
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.pca = PCA(n_components=n_comp)
        X_s = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_s)

        for cls in np.unique(y):
            mask = y == cls
            cls_data = X_pca[mask]
            mean = cls_data.mean(axis=0)
            if cls_data.shape[0] > 1:
                cov = np.cov(cls_data.T)
            else:
                cov = np.eye(n_comp)
            cov += 1e-4 * np.eye(n_comp)
            cov_inv = np.linalg.inv(cov)
            self.class_stats[cls] = (mean, cov_inv)

        # 归一化用中位数
        dists_train = []
        for i in range(len(y)):
            mean, cov_inv = self.class_stats[y[i]]
            diff = X_pca[i] - mean
            dists_train.append(np.sqrt(max(0, diff @ cov_inv @ diff)))
        self._med_dist = max(np.median(dists_train), 1e-6)

    def predict(self, X):
        X_s = self.scaler.transform(X)
        X_pca = self.pca.transform(X_s)
        preds, scores = [], []
        for x in X_pca:
            best_cls, best_dist = None, np.inf
            for cls, (mean, cov_inv) in self.class_stats.items():
                diff = x - mean
                dist = np.sqrt(max(0, diff @ cov_inv @ diff))
                if dist < best_dist:
                    best_dist = dist
                    best_cls = cls
            preds.append(best_cls)
            scores.append(float(np.exp(-best_dist / self._med_dist)))
        return np.array(preds), np.array(scores)

    def get_embeddings(self, X):
        X_s = self.scaler.transform(X)
        return self.pca.transform(X_s)


# ─────────────────────────────────────────────────────────
#  PLS-DA
# ─────────────────────────────────────────────────────────

class PLSDABaseline(TraditionalBaseline):
    """PCA 预处理 + PLS-DA 分类。"""

    def __init__(self, n_components=10, n_pca=100):
        self.n_components = n_components
        self.n_pca = n_pca
        self.scaler = StandardScaler()
        self.pca = None
        self.pls = None
        self.classes_ = None

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        n_pca = min(self.n_pca, X.shape[0] - 1, X.shape[1])
        self.pca = PCA(n_components=n_pca)
        X_s = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_s)

        # One-hot 编码 Y
        Y = np.zeros((len(y), n_classes))
        for i, cls in enumerate(self.classes_):
            Y[y == cls, i] = 1.0

        n_comp = min(self.n_components, X_pca.shape[1], n_classes)
        self.pls = PLSRegression(n_components=max(1, n_comp))
        self.pls.fit(X_pca, Y)

    def predict(self, X):
        X_s = self.scaler.transform(X)
        X_pca = self.pca.transform(X_s)
        Y_pred = self.pls.predict(X_pca)

        preds = self.classes_[Y_pred.argmax(axis=1)]
        # Softmax 归一化作为置信度
        exp_y = np.exp(Y_pred - Y_pred.max(axis=1, keepdims=True))
        probs = exp_y / exp_y.sum(axis=1, keepdims=True)
        scores = probs.max(axis=1)
        return preds, scores

    def get_embeddings(self, X):
        X_s = self.scaler.transform(X)
        X_pca = self.pca.transform(X_s)
        return self.pls.transform(X_pca)


# ─────────────────────────────────────────────────────────
#  SVM-RBF
# ─────────────────────────────────────────────────────────

class SVMBaseline(TraditionalBaseline):
    """PCA + SVM (RBF kernel, Platt calibration)。"""

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = None
        self.svm = None

    def fit(self, X, y):
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.pca = PCA(n_components=n_comp)
        X_s = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_s)
        self.svm = SVC(kernel="rbf", probability=True, random_state=42)
        self.svm.fit(X_pca, y)

    def predict(self, X):
        X_s = self.scaler.transform(X)
        X_pca = self.pca.transform(X_s)
        preds = self.svm.predict(X_pca)
        probs = self.svm.predict_proba(X_pca)
        scores = probs.max(axis=1)
        return preds, scores

    def get_embeddings(self, X):
        X_s = self.scaler.transform(X)
        return self.pca.transform(X_s)


# ─────────────────────────────────────────────────────────
#  Random Forest
# ─────────────────────────────────────────────────────────

class RandomForestBaseline(TraditionalBaseline):
    """PCA + Random Forest。"""

    def __init__(self, n_components=50, n_estimators=200):
        self.n_components = n_components
        self.n_estimators = n_estimators
        self.scaler = StandardScaler()
        self.pca = None
        self.rf = None

    def fit(self, X, y):
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.pca = PCA(n_components=n_comp)
        X_s = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_s)
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=42, n_jobs=-1
        )
        self.rf.fit(X_pca, y)

    def predict(self, X):
        X_s = self.scaler.transform(X)
        X_pca = self.pca.transform(X_s)
        preds = self.rf.predict(X_pca)
        probs = self.rf.predict_proba(X_pca)
        scores = probs.max(axis=1)
        return preds, scores

    def get_embeddings(self, X):
        X_s = self.scaler.transform(X)
        return self.pca.transform(X_s)


# ═══════════════════════════════════════════════════════════
#  深度学习对比: 无双轴注意力编码器
# ═══════════════════════════════════════════════════════════

class PlainEncoder(nn.Module):
    """标准 ResNet 编码器 (无双轴注意力), 接口与 GCMSEncoder 一致。"""

    def __init__(self, in_channels=2, channels=(32, 64, 128, 256),
                 dropout=0.3, blocks_per_stage=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = _make_stage(channels[0], channels[1], blocks_per_stage, stride=2)
        self.stage2 = _make_stage(channels[1], channels[2], blocks_per_stage, stride=2)
        self.stage3 = _make_stage(channels[2], channels[3], blocks_per_stage, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.out_dim = channels[3]

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feat_map = x
        z = self.pool(x).flatten(1)
        z = self.drop(z)
        return z, feat_map


# ═══════════════════════════════════════════════════════════
#  DL 基线模型
# ═══════════════════════════════════════════════════════════

class BaselineCNN(nn.Module):
    """
    DL 基线: PlainEncoder + 分类头。
    embed_normalize=True 用于度量学习方法 (Triplet, Center)。
    """

    def __init__(self, num_classes, cfg, embed_normalize=False):
        super().__init__()
        self.embed_normalize = embed_normalize
        self.encoder = PlainEncoder(
            in_channels=cfg.in_channels,
            channels=cfg.encoder_channels,
            dropout=cfg.dropout,
            blocks_per_stage=cfg.blocks_per_stage,
        )
        dim = self.encoder.out_dim
        self.cls_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        z_raw, feat_map = self.encoder(x)
        z = F.normalize(z_raw, dim=1) if self.embed_normalize else z_raw
        logits = self.cls_head(z_raw)
        return {"z": z, "z_raw": z_raw, "logits": logits}

    def encode(self, x):
        z_raw, _ = self.encoder(x)
        return F.normalize(z_raw, dim=1) if self.embed_normalize else z_raw


# ═══════════════════════════════════════════════════════════
#  DL 基线损失函数
# ═══════════════════════════════════════════════════════════

class TripletLoss(nn.Module):
    """在线难样本挖掘三元组损失。"""

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        device = embeddings.device
        B = len(embeddings)
        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        dists = torch.cdist(embeddings.unsqueeze(0),
                            embeddings.unsqueeze(0)).squeeze(0)  # (B, B)

        loss = torch.tensor(0.0, device=device)
        count = 0
        for i in range(B):
            pos_mask = (labels == labels[i])
            pos_mask[i] = False
            neg_mask = (labels != labels[i])

            if not pos_mask.any() or not neg_mask.any():
                continue

            hardest_pos = dists[i][pos_mask].max()
            hardest_neg = dists[i][neg_mask].min()

            loss = loss + F.relu(hardest_pos - hardest_neg + self.margin)
            count += 1

        return loss / max(count, 1)


class CenterLoss(nn.Module):
    """中心损失 (Wen et al. 2016): 拉近样本到其类中心。"""

    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x, labels):
        centers_batch = self.centers[labels]
        return ((x - centers_batch) ** 2).sum(dim=1).mean()


class BaselineLoss(nn.Module):
    """DL 基线组合损失: ce / triplet+ce / center+ce。"""

    def __init__(self, method, num_classes=None, feat_dim=None):
        super().__init__()
        self.method = method
        self.ce = nn.CrossEntropyLoss()

        if method == "triplet":
            self.triplet = TripletLoss(margin=0.5)
            self.lam_triplet = 1.0
        elif method == "center":
            assert num_classes is not None and feat_dim is not None
            self.center = CenterLoss(num_classes, feat_dim)
            self.lam_center = 0.01

    def forward(self, model_out, batch):
        losses = {}
        losses["cls"] = self.ce(model_out["logits"], batch["product"])

        if self.method == "triplet":
            losses["triplet"] = self.triplet(model_out["z"], batch["product"])
            losses["total"] = losses["cls"] + self.lam_triplet * losses["triplet"]
        elif self.method == "center":
            losses["center"] = self.center(model_out["z_raw"], batch["product"])
            losses["total"] = losses["cls"] + self.lam_center * losses["center"]
        else:
            losses["total"] = losses["cls"]

        return losses


# ═══════════════════════════════════════════════════════════
#  DL 基线训练
# ═══════════════════════════════════════════════════════════

def train_baseline_epoch(model, loader, criterion, optimizer, device):
    """DL 基线单 epoch 训练。"""
    model.train()
    running = {}
    for batch in loader:
        x = batch["input"].to(device)
        batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        out = model(x)
        losses = criterion(out, batch_dev)

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + v.item()

    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def validate_baseline(model, loader, criterion, device):
    """DL 基线验证。"""
    model.eval()
    running = {}
    all_pred, all_true = [], []
    for batch in loader:
        x = batch["input"].to(device)
        batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        out = model(x)
        losses = criterion(out, batch_dev)

        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + v.item()
        all_pred.append(out["logits"].argmax(dim=1).cpu())
        all_true.append(batch["product"])

    n = max(len(loader), 1)
    metrics = {k: v / n for k, v in running.items()}
    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    metrics["acc"] = (all_pred == all_true).float().mean().item()
    return metrics


def train_dl_baseline_fold(method_name, train_idx, val_idx, batch_name,
                           metadata_csv, cfg):
    """训练一个 DL 基线方法的单 fold, 返回 (model, ds_train, ds_val, loader_val)。"""
    from train import build_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")

    ds_train, ds_val, loader_train, loader_val = build_loaders(
        metadata_csv, train_idx, val_idx, cfg, product_col
    )

    num_classes = ds_train.num_products
    method_map = {
        "ResNet-CE": "ce",
        "ResNet-Triplet": "triplet",
        "ResNet-Center": "center",
    }
    method = method_map[method_name]
    embed_norm = method in ("triplet", "center")

    model = BaselineCNN(num_classes, cfg, embed_normalize=embed_norm).to(device)
    feat_dim = model.encoder.out_dim
    criterion = BaselineLoss(method, num_classes, feat_dim).to(device)

    # 训练总 epoch = pretrain + finetune (公平比较)
    total_epochs = cfg.epochs_pretrain + cfg.epochs_finetune
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(
        all_params, lr=cfg.lr_finetune, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs
    )

    best_acc = 0
    best_state = None

    for epoch in range(total_epochs):
        train_baseline_epoch(model, loader_train, criterion, optimizer, device)
        m_val = validate_baseline(model, loader_val, criterion, device)
        scheduler.step()

        if m_val["acc"] > best_acc:
            best_acc = m_val["acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 40 == 0:
            print(f"    Epoch {epoch+1}/{total_epochs} "
                  f"val_acc={m_val['acc']:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, ds_train, ds_val, loader_val
