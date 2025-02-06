import numpy as np
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
from pypots.data import load_specific_dataset
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Tuple

# data = load_specific_dataset('physionet_2012')  # PyPOTS will automatically download and extract it.
data = pd.read_csv('raw_data.csv',nrows=13104)
# X = data['X']
# num_samples = len(X['RecordID'].unique())
data = data.drop(['id', 'power_p'], axis = 1)
X = StandardScaler().fit_transform(data.to_numpy())
X = X.reshape(273, 48, -1)
X_ori = X  # keep X_ori for validation
X = mcar(X, 0.3)  # randomly hold out 10% observed values as ground truth
dataset = {"X": X}  # X for model input
print(X.shape)  # (2730, 48, 11), 11988 samples and each sample has 48 time steps, 37 features
myX = np.nan_to_num(X, nan=-1)
Xdata = torch.tensor(myX)
X_oridata = torch.tensor(X_ori)

# pmask = Xdata.clone().float()
# # 指定的值
# specified_value = -1
#
# # 将指定值变成 0，其余值变成 1
# mask = torch.where(pmask == specified_value, torch.tensor(0), torch.tensor(1))

# 示例数据：包含缺失值的时间序列
batch_size = 273
seq_len = 48
input_dim = 11

input_data = Xdata.clone().float()

input_data[input_data == -1] = 0  # 将缺失值替换为0或其他填充值
mask = torch.ones(batch_size, seq_len, input_dim).float()
target_data = X_oridata.clone().float()  # 目标数据用于计算损失

# Model training. This is PyPOTS showtime.    SAITS模型
from pypots.imputation import SAITS
from pypots.utils.metrics import calc_mae
from pypots.utils.metrics import calc_mse
from pypots.utils.metrics import calc_mre
from pypots.utils.metrics import calc_rmse



class AttentionOperator(nn.Module):
    """
    The abstract class for all attention layers.
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class CosFormerAttention(AttentionOperator):
    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Cosine similarity
        attn = F.cosine_similarity(q.unsqueeze(-2), k.unsqueeze(-3), dim=-1) / self.temperature

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            attn_opt: AttentionOperator,
            d_model: int,
            n_heads: int,
            d_k: int,
            d_v: int,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=False)

        self.attention_operator = attn_opt
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, q_len = q.size(0), q.size(1)
        k_len = k.size(1)
        v_len = v.size(1)

        q = self.w_qs(q).view(batch_size, q_len, self.n_heads, self.d_k)
        k = self.w_ks(k).view(batch_size, k_len, self.n_heads, self.d_k)
        v = self.w_vs(v).view(batch_size, v_len, self.n_heads, self.d_v)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)

        v, attn_weights = self.attention_operator(q, k, v, attn_mask, **kwargs)
        v = v.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        v = self.fc(v)

        return v, attn_weights


class CosFormerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(
            CosFormerAttention(temperature=d_k ** 0.5, attn_dropout=dropout),
            d_model,
            n_heads,
            d_k,
            d_v,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x


class CosFormerImputer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int, num_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            CosFormerLayer(d_model, n_heads, d_k, d_v, d_ff, dropout)
            for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(self, x, attn_mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.output_layer(x)
        return x


# 定义模型
model = CosFormerImputer(input_dim=input_dim, d_model=256, n_heads=4, d_k=64, d_v=64, d_ff=128, num_layers=2)

# 定义损失函数和优化器
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    output = model(input_data)

    # 只计算非缺失值的损失
    loss = criterion(output * mask, target_data * mask)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 插补后的输出
model.eval()
with torch.no_grad():
    imputed_data = model(input_data)


imputation = imputed_data.detach().cpu().numpy()
indicating_mask = np.isnan(X) ^ np.isnan(X_ori)  # indicating mask for imputation error calculation
mae = calc_mae(imputation, np.nan_to_num(X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
print(mae)
mse = calc_mse(imputation, np.nan_to_num(X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
print(mse)
mre = calc_mre(imputation, np.nan_to_num(X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
print(mre)
rmse = calc_rmse(imputation, np.nan_to_num(X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
print(rmse)
