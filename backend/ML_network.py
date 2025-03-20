import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, classification_report)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------数据预处理--------------------
data = pd.read_csv(r"C:\Users\99569\Desktop\ML-group\diabetes.csv")

# 处理0值（糖尿病数据中的0可能表示缺失值）
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[zero_features] = data[zero_features].replace(0, np.nan)
for col in zero_features:
    data[col].fillna(data[col].median(), inplace=True)

# 特征/标签分离
X = data.drop('Outcome', axis=1).values
y =data['Outcome'].values

# 转换为Tensor
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).reshape(-1, 1)

# --------------------定义Pipeline组件--------------------
class TorchPreprocessor(TransformerMixin, BaseEstimator):
    """自定义预处理组件"""
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        # 确保输入是NumPy数组
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        # 确保输入是NumPy数组
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        X_scaled = self.scaler.transform(X)
        return torch.FloatTensor(X_scaled)

class TorchEstimator(BaseEstimator):
    """PyTorch模型封装器"""
    def __init__(self, hidden_layers=[64,32], dropout=0.5, lr=0.001, epochs=100):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        
    def fit(self, X, y):
        # 确保输入是Tensor
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.FloatTensor(y).reshape(-1, 1)
        
        # 初始化模型
        self.model = DiabetesMLP(
            input_size=X.shape[1],
            hidden_layers=self.hidden_layers,
            dropout=self.dropout
        ).to(device)
        
        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        
        # 转换为DataLoader
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 训练循环
        for _ in range(self.epochs):
            self.model.train()
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return self
    
    def predict_proba(self, X):
        # 确保输入是Tensor
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X.to(device)).cpu().numpy()
        return np.concatenate([1-preds, preds], axis=1)

# 定义MLP模型
class DiabetesMLP(nn.Module):
    def __init__(self, input_size=8, hidden_layers=[64, 32], dropout=0.5):
        super(DiabetesMLP, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.net(x))

# --------------------构建完整Pipeline--------------------
full_pipeline = Pipeline([
    ('preprocessor', TorchPreprocessor()),
    ('estimator', TorchEstimator())
])

# --------------------交叉验证评估--------------------
def cross_validate_pipeline(pipeline, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    accuracies = []
    auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'\nFold {fold+1}/{n_splits}')
        
        # 获取原始数据划分
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 克隆pipeline避免参数泄露
        cloned_pipe = clone(pipeline)
        
        # 训练（自动执行预处理）
        cloned_pipe.fit(X_train, y_train)
        
        # 预测
        preds = cloned_pipe.predict_proba(X_val)[:, 1]
        preds_class = (preds >= 0.5).astype(float)
        
        # 评估
        acc = accuracy_score(y_val, preds_class)
        auc = roc_auc_score(y_val, preds)
        
        accuracies.append(acc)
        auc_scores.append(auc)
    
    print(f"\nCross-Validation Performance:")
    print(f"Average Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
    print(f"Average AUC: {np.mean(auc_scores):.4f} (±{np.std(auc_scores):.4f})")
    return np.mean(auc_scores)

# 执行交叉验证
base_score = cross_validate_pipeline(full_pipeline, X, y)

# --------------------超参数优化--------------------
param_grid = {
    'estimator__hidden_layers': [[64,32], [128,64]],
    'estimator__dropout': [0.3, 0.5],
    'estimator__lr': [0.001, 0.0005],
    'estimator__epochs': [100]
}

grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=2
)

# 转换数据格式以适应GridSearchCV
X_np = X.numpy()
y_np = y.numpy().ravel()

grid_search.fit(X_np, y_np)

# --------------------最终评估与保存--------------------
# 划分原始数据集
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    data.drop('Outcome', axis=1).values, 
    data['Outcome'].values,
    test_size=0.2, 
    stratify=data['Outcome'],
    random_state=42
)

# 使用最佳pipeline训练
best_pipeline = grid_search.best_estimator_
best_pipeline.fit(X_train_raw, y_train)

# 测试评估
test_preds_proba = best_pipeline.predict_proba(X_test_raw)[:, 1]
test_preds_class = (test_preds_proba >= 0.5).astype(float)

# 计算评估指标
accuracy = accuracy_score(y_test, test_preds_class)
precision = precision_score(y_test, test_preds_class)
recall = recall_score(y_test, test_preds_class)
f1 = f1_score(y_test, test_preds_class)
auc = roc_auc_score(y_test, test_preds_proba)

# 计算特异性
tn, fp, fn, tp = confusion_matrix(y_test, test_preds_class).ravel()
specificity = tn / (tn + fp)

# 打印评估结果
print(f"\nFinal Test Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, test_preds_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 保存完整pipeline
dump(best_pipeline, 'best_pipeline.joblib')