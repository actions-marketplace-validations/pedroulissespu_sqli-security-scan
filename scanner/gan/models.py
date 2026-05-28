import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Gerador baseado em LSTM que mapeia vetores de ruído latente
    para sequências de payloads SQL Injection.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, max_len, num_layers=2, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass do gerador.
        x: (batch, seq_len) - índices de tokens
        """
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, hidden = self.lstm(embedded, hidden)  # (batch, seq_len, hidden_dim)
        logits = self.fc_out(output)  # (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)


class Discriminator(nn.Module):
    """
    Discriminador baseado em LSTM que classifica sequências como
    reais (dos datasets) ou sintéticas (geradas pelo Generator).
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Forward pass do discriminador.
        x: (batch, seq_len) - índices de tokens
        """
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim)

        # Usar a saída do último time step
        last_output = output[:, -1, :]  # (batch, hidden_dim)

        prob = self.classifier(last_output)  # (batch, 1)
        return prob.squeeze(-1)
