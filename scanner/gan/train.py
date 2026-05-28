import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scanner.gan.models import Generator, Discriminator
from scanner.gan.preprocessing import (
    prepare_training_data,
    save_vocab,
    SOS_TOKEN,
)


# Configurações padrão
DEFAULT_CONFIG = {
    "embed_dim": 128,
    "hidden_dim": 512,
    "num_layers": 3,
    "dropout": 0.3,
    "max_len": 256,
    "max_vocab_size": 256,
    "batch_size": 256,
    "epochs": 500,
    "lr_gen": 1e-4,
    "lr_disc": 3e-4,
    "max_samples_per_dataset": None,
    "teacher_forcing_ratio": 0.5,
    "label_smoothing": 0.1,
}


def _teacher_forcing_step(generator, real_data, criterion, device):
    """Treina o gerador com teacher forcing (pré-treino / loss auxiliar)."""
    batch_size, seq_len = real_data.shape

    # Input: todos os tokens exceto o último
    input_seq = real_data[:, :-1]
    # Target: todos os tokens exceto o primeiro
    target_seq = real_data[:, 1:]

    logits, _ = generator(input_seq)
    # Reshape para CrossEntropy: (batch * seq, vocab) vs (batch * seq)
    loss = criterion(
        logits.reshape(-1, generator.vocab_size),
        target_seq.reshape(-1),
    )
    return loss


def _generate_fake_sequences(generator, batch_size, max_len, vocab, device):
    """Gera sequências falsas autoregressivamente."""
    sos_idx = vocab["char2idx"][SOS_TOKEN]

    # Iniciar com token SOS
    current = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    hidden = generator.init_hidden(batch_size, device)

    generated = [current]

    for _ in range(max_len - 1):
        logits, hidden = generator(current, hidden)
        # Amostragem com softmax
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, 1)
        generated.append(next_token)
        current = next_token

    return torch.cat(generated, dim=1)  # (batch, max_len)


def train_gan(datasets_dir="datasets", output_dir="models", config=None, resume_path=None):
    """Loop principal de treinamento da GAN."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Treino] Dispositivo: {device}")

    # Preparar dados
    print("[Treino] Carregando e pré-processando datasets...")
    real_data, benign_data, vocab = prepare_training_data(
        max_len=cfg["max_len"],
        max_vocab_size=cfg["max_vocab_size"],
        datasets_dir=datasets_dir,
        max_samples_per_dataset=cfg["max_samples_per_dataset"],
    )

    vocab_size = vocab["vocab_size"]

    # DataLoader
    dataset = TensorDataset(real_data)
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    # Instanciar modelos
    generator = Generator(
        vocab_size=vocab_size,
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        max_len=cfg["max_len"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    discriminator = Discriminator(
        vocab_size=vocab_size,
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    # Otimizadores
    opt_gen = torch.optim.Adam(generator.parameters(), lr=cfg["lr_gen"], betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=cfg["lr_disc"], betas=(0.5, 0.999))

    # Carregar checkpoint se resume
    start_epoch = 1
    if resume_path:
        print(f"[Treino] Retomando de: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint["generator_state"])
        discriminator.load_state_dict(checkpoint["discriminator_state"])
        opt_gen.load_state_dict(checkpoint["opt_gen_state"])
        opt_disc.load_state_dict(checkpoint["opt_disc_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"[Treino] Continuando da época {start_epoch} (D_loss: {checkpoint['d_loss']:.4f}, G_loss: {checkpoint['g_loss']:.4f})")

    # Losses
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # ignora PAD

    # Label smoothing
    real_label = 1.0 - cfg["label_smoothing"]
    fake_label = cfg["label_smoothing"]

    os.makedirs(output_dir, exist_ok=True)

    print(f"[Treino] Iniciando {cfg['epochs']} épocas (a partir da {start_epoch})...")
    start_time = time.time()

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0

        for (batch_real,) in dataloader:
            batch_real = batch_real.to(device)
            batch_size_actual = batch_real.shape[0]

            # === Treinar Discriminador ===
            opt_disc.zero_grad()

            # Dados reais
            real_pred = discriminator(batch_real)
            real_labels = torch.full((batch_size_actual,), real_label, device=device)
            d_loss_real = bce_loss(real_pred, real_labels)

            # Dados falsos
            with torch.no_grad():
                fake_data = _generate_fake_sequences(
                    generator, batch_size_actual, cfg["max_len"], vocab, device
                )
            fake_pred = discriminator(fake_data)
            fake_labels = torch.full((batch_size_actual,), fake_label, device=device)
            d_loss_fake = bce_loss(fake_pred, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            opt_disc.step()

            # === Treinar Gerador ===
            opt_gen.zero_grad()

            # Loss adversarial: enganar o discriminador
            fake_data_g = _generate_fake_sequences(
                generator, batch_size_actual, cfg["max_len"], vocab, device
            )
            fake_pred_g = discriminator(fake_data_g)
            g_loss_adv = bce_loss(fake_pred_g, torch.ones(batch_size_actual, device=device))

            # Loss de teacher forcing (auxiliar)
            g_loss_tf = _teacher_forcing_step(generator, batch_real, ce_loss, device)

            g_loss = g_loss_adv + cfg["teacher_forcing_ratio"] * g_loss_tf
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_gen.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1

        avg_d = epoch_d_loss / max(num_batches, 1)
        avg_g = epoch_g_loss / max(num_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"  Época {epoch:3d}/{cfg['epochs']} | D_loss: {avg_d:.4f} | G_loss: {avg_g:.4f} | {elapsed:.1f}s")

        # Checkpoint intermediário a cada 50 épocas
        if epoch % 50 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "generator_state": generator.state_dict(),
                "discriminator_state": discriminator.state_dict(),
                "opt_gen_state": opt_gen.state_dict(),
                "opt_disc_state": opt_disc.state_dict(),
                "config": cfg,
                "vocab": {
                    "char2idx": vocab["char2idx"],
                    "idx2char": {str(k): v for k, v in vocab["idx2char"].items()},
                    "vocab_size": vocab["vocab_size"],
                    "max_len": vocab["max_len"],
                },
                "d_loss": avg_d,
                "g_loss": avg_g,
            }, checkpoint_path)
            print(f"  [Checkpoint] Salvo em: {checkpoint_path}")

    # Salvar modelo final
    total_time = time.time() - start_time
    print(f"\n[Treino] Concluído em {total_time:.1f}s")

    model_path = os.path.join(output_dir, "gan_sqli.pt")
    torch.save({
        "generator_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "config": cfg,
        "vocab": {
            "char2idx": vocab["char2idx"],
            "idx2char": {str(k): v for k, v in vocab["idx2char"].items()},
            "vocab_size": vocab["vocab_size"],
            "max_len": vocab["max_len"],
        },
    }, model_path)
    print(f"[Treino] Modelo salvo em: {model_path}")

    # Salvar vocab separado
    vocab_path = os.path.join(output_dir, "vocab.json")
    save_vocab(vocab, vocab_path)

    return model_path
