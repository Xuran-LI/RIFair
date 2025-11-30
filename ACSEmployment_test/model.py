import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ============================================================
# MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x_value):
        x = x_value
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.out(x), dim=-1)
        return x

# ============================================================
# Wide & Deep
# ============================================================
class WideDeep(nn.Module):
    def __init__(self, num_fields, vocab_size=100, emb_dim=8, num_classes=2):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        # -------- Shared Embedding Layer --------
        self.emb_layers = nn.Embedding(vocab_size, emb_dim)
        # -------- Wide Component --------
        # 输入维度 = num_fields
        self.wide = nn.Linear(num_fields, 1, bias=False)
        # -------- Deep Component --------
        # Deep 输入维度 = num_fields * emb_dim
        self.deep_fc1 = nn.Linear(num_fields * emb_dim, 128)
        self.deep_fc2 = nn.Linear(128, 64)
        self.deep_out = nn.Linear(64, 1)
        # -------- Final --------
        self.final = nn.Linear(1, num_classes)

    def forward(self, x_index, x_value):
        """
        x_index : [B, F]  long
        x_value : [B, F]  float
        """
        # =====================
        # Wide = linear(x_value)
        # =====================
        wide_out = self.wide(x_value)     # [B, 1]
        # =====================
        # Deep
        # =====================
        deep_emb = self.emb_layers(x_index)   # [B, F, emb_dim]
        # Flatten for Linear
        deep_in = deep_emb.view(deep_emb.size(0), -1)  # [B, F * emb_dim]
        x = F.relu(self.deep_fc1(deep_in))
        x = F.relu(self.deep_fc2(x))
        deep_out = self.deep_out(x)                    # [B, 1]
        # =====================
        # Combine
        # =====================
        out = wide_out + deep_out                     # [B, 1]
        # =====================
        # Final logits
        # =====================
        logits = self.final(out)                      # [B, num_classes]
        return logits

# ============================================================
# DeepFM
# ============================================================
class DeepFM(nn.Module):
    def __init__(self, num_fields, vocab_size=100, emb_dim=8, num_classes=2):
        super().__init__()

        self.num_fields = num_fields
        self.emb_dim = emb_dim

        # --- Shared embeddings ---
        self.first_order_emb = nn.Embedding(vocab_size, 1)
        self.second_order_emb = nn.Embedding(vocab_size, emb_dim)

        # --- Deep component ---
        self.deep_fc1 = nn.Linear(num_fields * emb_dim, 128)
        self.deep_fc2 = nn.Linear(128, 64)
        self.deep_out = nn.Linear(64, 1)

        # final classifier
        self.final = nn.Linear(1, num_classes)

    def forward(self, x_index, x_value):
        """
        x_index: [B, F]  Long
        x_value: [B, F]  Float
        """
        # =====================
        # 1st-order part
        # =====================
        first_order = self.first_order_emb(x_index)  # [B, F, 1]
        first_order = first_order.squeeze(-1) * x_value  # [B, F]
        first_order = torch.sum(first_order, dim=1, keepdim=True)  # [B, 1]

        # =====================
        # 2nd-order part
        # =====================
        v = self.second_order_emb(x_index)  # [B, F, K]
        v = v * x_value.unsqueeze(-1)  # [B, F, K]
        sum_v = torch.sum(v, dim=1)  # [B, K]
        sum_v_square = sum_v ** 2  # [B, K]
        square_v = v ** 2  # [B, F, K]
        square_sum_v = torch.sum(square_v, dim=1)  # [B, K]
        second_order = 0.5 * torch.sum(sum_v_square - square_sum_v, dim=1, keepdim=True)  # [B, 1]

        # =====================
        # Deep part
        # =====================
        deep_in = v.view(v.size(0), -1)  # [B, F*K]
        x = F.relu(self.deep_fc1(deep_in))
        x = F.relu(self.deep_fc2(x))
        deep_out = self.deep_out(x)  # [B, 1]
        # =====================
        # Combine FM + Deep
        # =====================
        combined = first_order + second_order + deep_out  # [B, 1]
        # =====================
        # Final classifier
        # =====================
        logits = self.final(combined)  # [B, num_classes]
        return logits

    # def forward(self, x_index, x_value):
    #     # First-order
    #     # first_order = torch.cat([self.first_order_emb[i](x_index[:, i + 1]) for i in range(self.num_fields - 1)],dim=-1)
    #     # first_order = first_order.squeeze(-1)
    #     # first_order = torch.cat([x_value[:, :1], first_order], dim=-1).sum(dim=-1, keepdim=True)
    #     first_order = self.first_order_emb(x_index)
    #
    #     # Second-order
    #     # second_order_embs = [self.second_order_emb[i](x_index[:, i + 1]) for i in range(self.num_fields - 1)]
    #     v = self.second_order_emb(x_index)  # [B, F, K]
    #     v = v * x_value.unsqueeze(-1)
    #     sum_v = torch.sum(v, dim=1)
    #     sum_v_square = sum_v ** 2
    #     square_sum_v = torch.sum(v ** 2, dim=1)
    #     second_order = 0.5 * torch.sum(sum_v_square - square_sum_v, dim=1, keepdim=True)
    #
    #     # ---------------- Deep ----------------
    #     deep_input = v.flatten(start_dim=1)
    #     x = F.relu(self.deep_fc1(deep_input))
    #     x = F.relu(self.deep_fc2(x))
    #     deep_out = self.deep_out(x)
    #
    #     # ---------------- Combine ----------------
    #     combined = first_order + second_order + deep_out
    #
    #     # ---------------- Final classifier ----------------
    #     logits = self.final(combined)  # [B, num_classes], **不要加 softmax**
    #     return logits


# ===============================
# ACSEmploymentEmbedding
# ===============================
class ACSEmploymentEmbedding(nn.Module):
    def __init__(self, vocab_size=100, embed_size=32, dropout_rate=0.1):
        super().__init__()
        self.index_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embed_index, embed_value):
        """
        embed_index: [batch, field_size], LongTensor
        embed_value: [batch, field_size], FloatTensor
        returns: [batch, field_size, embed_size]
        """
        x = self.index_emb(embed_index)  # [B, F, E]
        embed_value = embed_value.unsqueeze(-1)  # [B, F, 1]
        x = x * embed_value  # broadcasting
        x = self.dropout(x)
        return x

# ===============================
# AutoInt Transformer Block
# ===============================
class AutoIntTransformerBlock(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=64, dropout_rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attn(x, x, x)  # [B, F, E]
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)

        # Feed-forward
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        x = self.norm2(x + ffn_output)
        return x

# ===============================
# AutoInt Model
# ===============================
class AutoInt(nn.Module):
    def __init__(self, vocab_size=100, embed_size=32, field_size=16, num_heads=4, ff_dim=64, num_classes=2, num_blocks=2, dropout_rate=0.1):
        super().__init__()
        self.embedding = ACSEmploymentEmbedding(vocab_size=vocab_size, embed_size=embed_size, dropout_rate=dropout_rate)
        self.blocks = nn.ModuleList([
            AutoIntTransformerBlock(embed_dim=embed_size, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc1 = nn.Linear(embed_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, index_input, value_input):
        """
        index_input: [B, F] LongTensor
        value_input: [B, F] FloatTensor
        """
        x = self.embedding(index_input, value_input)  # [B, F, E]

        for block in self.blocks:
            x = block(x)  # [B, F, E]

        # Global average pooling
        x = x.transpose(1, 2)  # [B, E, F] for AdaptiveAvgPool1d
        x = self.pool(x).squeeze(-1)  # [B, E]

        # Classification head
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return F.softmax(x, dim=-1)

# ===============================
# TabTransformer Block
# ===============================
class TabTransformerBlock(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=128, dropout_rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

# ===============================
# TabTransformer Model
# ===============================
class TabTransformer(nn.Module):
    def __init__(self, num_fields, embed_dim=32, num_heads=4, ff_dim=128, num_classes=2, num_blocks=2, dropout_rate=0.1):
        super().__init__()
        # Tokenizer: map scalar -> embedding vector
        self.tokenizer = nn.Linear(num_fields, num_fields * embed_dim)
        self.embed_dim = embed_dim
        self.num_fields = num_fields
        self.layernorm0 = nn.LayerNorm(embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TabTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])

        # Classification head
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling over features
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc_out = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        x: [B, num_fields], FloatTensor
        """
        x = self.tokenizer(x)  # [B, num_fields * embed_dim]
        x = x.view(-1, self.num_fields, self.embed_dim)  # [B, num_fields, embed_dim]
        x = self.layernorm0(x)

        for block in self.blocks:
            x = block(x)  # [B, num_fields, embed_dim]

        # Global average pooling
        x = x.transpose(1, 2)  # [B, embed_dim, num_fields]
        x = self.pool(x).squeeze(-1)  # [B, embed_dim]

        # Classification head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return F.softmax(x, dim=-1)

# ============================================================
# Dataset wrapper
# ============================================================
class TabularDataset(Dataset):
    def __init__(self, index_data, value_data, labels):
        """
        index_data: [N, num_fields] int64
        value_data: [N, num_fields] float32
        labels: [N, num_classes] one-hot float32
        """
        self.index_data = torch.tensor(index_data, dtype=torch.long) if index_data is not None else None
        self.value_data = torch.tensor(value_data, dtype=torch.float32) if value_data is not None else None
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.index_data is not None and self.value_data is not None:
            return self.index_data[idx], self.value_data[idx], self.labels[idx]
        else:
            return self.value_data[idx], self.labels[idx]
