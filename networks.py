import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class StockDataset(Dataset):
    def __init__(self, data, target_col_idx, context_col_indices, seq_length=10):
        """
        Clase para transformar datos tabulares en ventanas temporales (secuencias 3D) para PyTorch.
        
        Parámetros:
        data (np.ndarray): Dataset escalado.
        target_col_idx (int): Índice de la columna a predecir (Retorno de JPM).
        context_col_indices (list): Índices de las columnas de contexto (BAC, C, WFC, XLF y sus indicadores).
        seq_length (int): Tamaño de la ventana histórica (ej. 10 días).
        """
        self.data = data
        self.target_col_idx = target_col_idx
        self.context_col_indices = context_col_indices
        self.seq_length = seq_length

    def __len__(self):
        # Cantidad de secuencias válidas
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Ventana de tiempo de tamaño seq_length
        window = self.data[idx : idx + self.seq_length]
        
        # Separar el objetivo y el contexto
        x_target = window[:, [self.target_col_idx]]  # (seq_length, 1) -> Mantenemos la 2D forma por día
        x_context = window[:, self.context_col_indices] # (seq_length, num_context_features)
        
        # El valor a predecir (T+1)
        y = self.data[idx + self.seq_length, self.target_col_idx]
        
        return (torch.tensor(x_target, dtype=torch.float32),
                torch.tensor(x_context, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, q, k, v):
        # Cross-Attention
        attn_output, attn_weights = self.cross_attention(query=q, key=k, value=v)
        
        # Residual + Norm
        x = self.norm1(q + attn_output)
        
        # FFN + Residual + Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x, attn_weights

class MasterStockTransformer(nn.Module):
    def __init__(self, target_features, context_features, d_model=64, nhead=4, num_layers=1, dropout=0.1):
        super(MasterStockTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Proyecciones lineales para llevar las features originales a la dimensión d_model
        self.target_proj = nn.Linear(target_features, d_model)
        self.context_proj = nn.Linear(context_features, d_model)
        
        # Múltiples bloques de Cross-Attention
        self.layers = nn.ModuleList([CrossAttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        
        # Predicción final
        self.predictor = nn.Linear(d_model, 1)
        
    def forward(self, x_target, x_context):
        # x_target: (Batch, Seq_Length, target_features)
        # x_context: (Batch, Seq_Length, context_features)
        
        q = self.target_proj(x_target)   # Query inicial: JPM Histórico
        k = self.context_proj(x_context) # Key: Mercado Histórico
        v = k                            # Value: Mercado Histórico
        
        # Pasamos la Query por las múltiples capas. Key y Value (mercado) se mantienen constantes.
        attn_weights = None
        for layer in self.layers:
            q, attn_weights = layer(q, k, v) # El output de la capa se vuelve la Query de la siguiente
        
        # Tomar solo la salida del último paso temporal de la secuencia para predecir el futuro
        last_timestep_output = q[:, -1, :]
        
        # Predicción escalar
        prediction = self.predictor(last_timestep_output)
        return prediction.squeeze(-1), attn_weights

class LSTMModel(nn.Module):
    def __init__(self, target_features, context_features, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=target_features + context_features, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        self.predictor = nn.Linear(hidden_size, 1)
        
    def forward(self, x_target, x_context):
        # El LSTM tradicional procesa todo junto, no separa en Query y Key.
        x = torch.cat((x_target, x_context), dim=-1)
        
        lstm_out, _ = self.lstm(x)
        
        # Tomar salida del último paso temporal
        last_timestep_output = lstm_out[:, -1, :]
        
        prediction = self.predictor(last_timestep_output)
        
        # Retornamos None para los attn_weights para ser compatibles con el training loop
        return prediction.squeeze(-1), None
