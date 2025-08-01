import torch
import torch.nn as nn
import math
from config import D_MODEL, DROPOUT_PROB, FRAME_SEQUENCE_LENGTH, N_HEADS, N_LAYERS, NUM_CLASSES, RAW_INPUT_SIZE
import config
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        cos_part = torch.cos(position * div_term)
        # Ensure slicing does not go out of bounds
        pe[:, 1::2] = cos_part[:, :pe[:, 1::2].size(1)]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, n_heads, n_layers, num_classes, dropout_prob):
        super(TransformerClassifier, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout_prob, max_len=FRAME_SEQUENCE_LENGTH)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout_prob, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # Use the output of the first token for classification
        output = output[:, 0, :]
        return self.fc(output)

def load_transformer_model(model_path):
    """Loads the transformer model and sets it to evaluation mode."""
    print("Loading Transformer model...")
    model = TransformerClassifier(RAW_INPUT_SIZE, D_MODEL, N_HEADS, N_LAYERS, NUM_CLASSES, DROPOUT_PROB).to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    return model