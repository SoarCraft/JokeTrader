import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, seq_feature_dim: int, account_dim: int, d_model: int = 128, nhead: int = 8, num_layers: int = 2):
        super(ActorCritic, self).__init__()
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输入嵌入投影: 将每个时间步的特征映射到d_model维
        self.feature_proj = nn.Linear(seq_feature_dim, d_model)
        # 可学习的位置编码（也可使用固定正弦位置编码）
        self.pos_embedding = nn.Parameter(torch.zeros(1, env.window_size+1, d_model))
        # 账户状态嵌入
        self.account_proj = nn.Linear(account_dim, d_model)
        # 融合后的全连接层
        self.fc_policy = nn.Linear(2 * d_model, 2)   # 输出动作均值(mu)两维 [position_frac_mu, leverage_mu]
        self.fc_value = nn.Linear(2 * d_model, 1)    # 输出状态价值V
        # 动作方差对数（作为全局参数，而非状态相关）
        self.log_std = nn.Parameter(torch.zeros(2))  # 两个动作维度各一个log_std参数
    
    def forward(self, history_seq, account_vec):
        # history_seq: [batch, seq_len, feature_dim]
        # account_vec: [batch, account_dim]
        batch_size = history_seq.size(0)
        seq_len = history_seq.size(1)
        # 对序列特征投影并加入位置编码
        x = self.feature_proj(history_seq)  # shape: [batch, seq_len, d_model]
        # 准备Transformer输入：可以在序列开头添加一个表示整体的CLS token
        # 我们用一个零初始化的cls_token参数表示
        cls_token = torch.zeros(batch_size, 1, x.size(2), device=x.device)
        # 将cls_token拼接到序列开头
        x = torch.cat([cls_token, x], dim=1)  # [batch, seq_len+1, d_model]
        # 加上位置编码
        x = x + self.pos_embedding[:, :seq_len+1, :]
        # Transformer编码
        x = x.transpose(0, 1)  # Transformer需要shape [seq_len+1, batch, d_model]
        encoded = self.transformer(x)       # 输出 shape: [seq_len+1, batch, d_model]
        encoded = encoded.transpose(0, 1)   # [batch, seq_len+1, d_model]
        # 取出序列第一个位置(CLS)的表示作为整体特征
        seq_repr = encoded[:, 0, :]         # [batch, d_model]
        # 账户状态映射
        acc_repr = torch.relu(self.account_proj(account_vec))  # [batch, d_model]
        # 融合两个表示
        fused = torch.cat([seq_repr, acc_repr], dim=-1)  # [batch, 2*d_model]
        # 输出策略的均值和状态价值
        policy_mean = self.fc_policy(torch.relu(fused))  # [batch, 2]
        value = self.fc_value(torch.relu(fused)).squeeze(-1)  # [batch]
        return policy_mean, value
