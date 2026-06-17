import re
import torch
from scanner.gan.models import Generator
from scanner.gan.preprocessing import decode_indices, SOS_TOKEN


# Padrões SQL mínimos que um payload válido deve conter
_VALID_SQL_RE = re.compile(
    r"(?:SELECT|INSERT|UPDATE|DELETE|DROP|UNION|WHERE|FROM|EXEC|CAST|"
    r"CONCAT|SLEEP|BENCHMARK|WAITFOR|INFORMATION_SCHEMA|LOAD_FILE|"
    r"CHR\s*\(|CHAR\s*\(|0x[0-9a-fA-F]{2,}|@@\w+|"
    r"OR\s+['\d]|AND\s+['\d]|"
    r"['\"];\s*--|['\"];|--\s*$|/\*|%27|%22|%3B|"
    r"'\s*(?:OR|AND)\s*['\d]|1\s*=\s*1|'\s*=\s*')",
    re.IGNORECASE,
)

# Caracteres inválidos excessivos (lixo do GAN)
_GARBAGE_RE = re.compile(r"[^\x20-\x7E]{3,}")

# Repetição excessiva de um mesmo char (ex: "aaaaaaa")
_REPETITION_RE = re.compile(r"(.)\1{7,}")


def _postprocess_payload(text):
    """Limpa e valida um payload gerado pelo GAN.

    Retorna o payload limpo ou None se for inválido/lixo.
    """
    if not text or len(text) < 5:
        return None

    # Remover espaços em excesso
    text = re.sub(r"\s+", " ", text).strip()

    # Descartar se tem blocos de lixo (caracteres não-ASCII repetidos)
    if _GARBAGE_RE.search(text):
        return None

    # Descartar se tem repetição excessiva (ex: "SSSSSSSSS")
    if _REPETITION_RE.search(text):
        return None

    # Descartar se não contém nenhum padrão SQL reconhecível
    if not _VALID_SQL_RE.search(text):
        return None

    # Balancear parênteses: truncar ou fechar
    open_count = text.count("(") - text.count(")")
    if open_count > 3:
        # Muitos parênteses abertos sem fechar — payload truncado demais
        text += ")" * min(open_count, 5)
    elif open_count < -2:
        # Mais fechamentos que aberturas — payload malformado
        return None

    # Balancear aspas simples (número ímpar = possivelmente truncado)
    if text.count("'") % 2 != 0:
        # Se termina sem fechar aspas, pode ser intencional (payload válido)
        # Só rejeita se a aspa solta está no meio sem contexto SQL
        pass

    # Remover trailing whitespace e comentários vazios
    text = text.rstrip()
    if text.endswith("/*"):
        text += "*/"
    if text.endswith("--"):
        text = text.rstrip("-").rstrip() + " --"

    return text if len(text) >= 5 else None


def _postprocess_batch(payloads):
    """Aplica pós-processamento a um lote de payloads. Retorna apenas os válidos."""
    result = []
    for p in payloads:
        cleaned = _postprocess_payload(p)
        if cleaned:
            result.append(cleaned)
    return result


# Prefixos de seed por tipo de parâmetro para guiar a geração contextual
_SEED_PREFIXES = {
    "integer": ["1 OR 1=1", "0 UNION SELECT", "-1 OR ", "1; DROP"],
    "number": ["1.0 OR 1=1", "0.0 UNION SELECT", "-1.0 OR "],
    "string": ["' OR '1'='1", "' UNION SELECT ", "'; DROP TABLE ", "\" OR \"1\"=\"1"],
    "default": ["' OR 1=1--", "1 UNION SELECT", "'; EXEC ", "' AND 1="],
}

# Estratégias baseadas na localização do parâmetro
_LOCATION_STRATEGIES = {
    "query": {"temperature": 0.7, "prefix_weight": 0.5},
    "path": {"temperature": 0.5, "prefix_weight": 0.7},
    "body": {"temperature": 0.8, "prefix_weight": 0.4},
    "header": {"temperature": 0.6, "prefix_weight": 0.3},
    "formData": {"temperature": 0.8, "prefix_weight": 0.4},
}


def load_trained_generator(model_path):
    """Carrega o gerador treinado a partir do checkpoint."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    vocab["idx2char"] = {int(k): v for k, v in vocab["idx2char"].items()}

    generator = Generator(
        vocab_size=vocab["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        max_len=config.get("max_len", 256),
        num_layers=config["num_layers"],
        dropout=0,  # inferência sem dropout
    )
    generator.load_state_dict(checkpoint["generator_state"])
    generator.eval()

    return generator, vocab, config


def _raw_generate(generator, vocab, config, num_payloads, temperature,
                  seed_prefix=None):
    """Geração bruta de sequências pelo gerador LSTM."""
    device = next(generator.parameters()).device
    max_len = config.get("max_len", 256)
    char2idx = vocab["char2idx"]
    idx2char = vocab["idx2char"]
    sos_idx = char2idx[SOS_TOKEN]

    payloads = []
    batch_size = min(num_payloads, 64)

    with torch.no_grad():
        remaining = num_payloads
        while remaining > 0:
            current_batch = min(batch_size, remaining)

            # Se há seed_prefix, alimentar o gerador com ele antes de gerar livremente
            if seed_prefix:
                seed_indices = [char2idx.get(ch, char2idx.get("<UNK>", 0))
                                for ch in seed_prefix]
                seed_seq = [sos_idx] + seed_indices
                current_input = torch.tensor(
                    [seed_seq] * current_batch, dtype=torch.long, device=device
                )
                hidden = generator.init_hidden(current_batch, device)
                _, hidden = generator(current_input, hidden)
                current = current_input[:, -1:]
                prefix_len = len(seed_seq)
            else:
                current = torch.full(
                    (current_batch, 1), sos_idx, dtype=torch.long, device=device
                )
                hidden = generator.init_hidden(current_batch, device)
                prefix_len = 1

            sequences = [current_input if seed_prefix else current]

            for _ in range(max_len - prefix_len):
                logits, hidden = generator(current, hidden)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                sequences.append(next_token)
                current = next_token

            generated = torch.cat(sequences, dim=1)

            for i in range(current_batch):
                seq = generated[i].tolist()
                text = decode_indices(seq, idx2char)
                text = text.strip()
                if text and len(text) > 3:
                    payloads.append(text)

            remaining -= current_batch

    return payloads


def generate_payloads(generator, vocab, config, num_payloads=200, temperature=0.7):
    """Gera payloads genéricos de SQL Injection (sem contexto de endpoint)."""
    # Gerar 2x mais para compensar os que serão filtrados no pós-processamento
    raw = _raw_generate(generator, vocab, config, num_payloads * 2, temperature)
    cleaned = _postprocess_batch(raw)
    unique = list(dict.fromkeys(cleaned))

    # Se não gerou o suficiente, tenta mais uma rodada com temperatura diferente
    if len(unique) < num_payloads:
        extra = _raw_generate(generator, vocab, config, num_payloads, temperature * 0.8)
        cleaned_extra = _postprocess_batch(extra)
        unique.extend(cleaned_extra)
        unique = list(dict.fromkeys(unique))

    return unique[:num_payloads]


def generate_payloads_for_endpoint(generator, vocab, config, endpoint,
                                   num_payloads=50, base_temperature=0.7):
    params = endpoint.get("params", [])
    if not params:
        return generate_payloads(generator, vocab, config, num_payloads, base_temperature)

    all_payloads = []

    # Agrupar parâmetros por localização
    param_groups = {}
    for p in params:
        loc = p.get("in", "query")
        if loc not in param_groups:
            param_groups[loc] = []
        param_groups[loc].append(p)

    # Para cada grupo de localização, gerar payloads adaptados
    payloads_per_group = max(num_payloads // len(param_groups), 10)

    for location, group_params in param_groups.items():
        strategy = _LOCATION_STRATEGIES.get(location, _LOCATION_STRATEGIES["query"])
        temp = base_temperature * strategy["temperature"]

        for param in group_params:
            param_type = param.get("type", "string")

            # Selecionar seeds pelo tipo do parâmetro
            seeds = _SEED_PREFIXES.get(param_type, _SEED_PREFIXES["default"])
            payloads_per_seed = max(payloads_per_group // (len(seeds) + 1), 5)

            # Geração com seed contextual (guiada pelo tipo)
            for seed in seeds:
                seeded = _raw_generate(
                    generator, vocab, config,
                    num_payloads=payloads_per_seed,
                    temperature=temp,
                    seed_prefix=seed,
                )
                all_payloads.extend(seeded)

            # Geração livre adicional (para diversidade)
            free = _raw_generate(
                generator, vocab, config,
                num_payloads=payloads_per_seed,
                temperature=base_temperature,
            )
            all_payloads.extend(free)

    # Pós-processamento: filtrar payloads inválidos e deduplicar
    cleaned = _postprocess_batch(all_payloads)
    unique = list(dict.fromkeys(cleaned))
    return unique[:num_payloads]
