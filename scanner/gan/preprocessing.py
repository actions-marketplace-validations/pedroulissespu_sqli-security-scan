import os
import random
import pandas as pd
import torch
import json
import re
import unicodedata


# Tokens especiais
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# Caracteres SQL relevantes para tokenização a nível de caractere
SQL_KEYWORDS = [
    "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "UNION", "WHERE",
    "FROM", "AND", "OR", "NOT", "NULL", "INTO", "VALUES", "SET",
    "ORDER", "BY", "GROUP", "HAVING", "LIKE", "IN", "BETWEEN",
    "EXISTS", "CREATE", "ALTER", "TABLE", "DATABASE", "EXEC",
    "EXECUTE", "CAST", "CONVERT", "CHAR", "VARCHAR", "CONCAT",
    "SUBSTRING", "ASCII", "SLEEP", "BENCHMARK", "WAITFOR", "DELAY",
    "INFORMATION_SCHEMA", "LOAD_FILE", "OUTFILE", "DUMPFILE",
]

_TEXT_REPLACEMENTS = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u00a0": " ",
})

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")
_WHITESPACE_RE = re.compile(r"\s+")
_PERCENT_HEX_RE = re.compile(r"%[0-9a-fA-F]{2}")
_HEX_LITERAL_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_SQL_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(sorted(SQL_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _tokenize_char_level(text, max_len):
    """Tokeniza o texto no nível de caractere."""
    tokens = list(text[:max_len])
    return tokens


def build_vocab(texts, max_vocab_size=256):
    """Constrói vocabulário baseado na frequência dos caracteres."""
    freq = {}
    for text in texts:
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1

    # Ordenar por frequência e limitar
    sorted_chars = sorted(freq.items(), key=lambda x: -x[1])
    vocab_chars = [ch for ch, _ in sorted_chars[:max_vocab_size - len(SPECIAL_TOKENS)]]

    # Construir mapeamentos
    char2idx = {}
    for i, token in enumerate(SPECIAL_TOKENS):
        char2idx[token] = i
    for i, ch in enumerate(vocab_chars):
        char2idx[ch] = i + len(SPECIAL_TOKENS)

    idx2char = {v: k for k, v in char2idx.items()}

    return char2idx, idx2char


def encode_text(text, char2idx, max_len):
    """Codifica texto em sequência de índices."""
    sos = char2idx[SOS_TOKEN]
    eos = char2idx[EOS_TOKEN]
    pad = char2idx[PAD_TOKEN]
    unk = char2idx.get(UNK_TOKEN, 0)

    tokens = _tokenize_char_level(text, max_len - 2)  # reservar SOS e EOS
    indices = [sos] + [char2idx.get(ch, unk) for ch in tokens] + [eos]

    # Padding
    while len(indices) < max_len:
        indices.append(pad)

    return indices[:max_len]


def decode_indices(indices, idx2char):
    """Decodifica sequência de índices de volta para texto."""
    chars = []
    for idx in indices:
        ch = idx2char.get(idx, "")
        if ch == EOS_TOKEN:
            break
        if ch in (PAD_TOKEN, SOS_TOKEN):
            continue
        chars.append(ch)
    return "".join(chars)


def _normalize_text(text):
    """Padroniza a representação textual para reduzir ruído e inconsistências."""
    if text is None:
        return None

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    normalized = normalized.translate(_TEXT_REPLACEMENTS)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()

    if not normalized:
        return None

    normalized = _PERCENT_HEX_RE.sub(lambda m: m.group(0).upper(), normalized)
    normalized = _HEX_LITERAL_RE.sub(
        lambda m: "0x" + m.group(0)[2:].upper(),
        normalized,
    )
    normalized = _SQL_KEYWORD_RE.sub(lambda m: m.group(1).upper(), normalized)
    return normalized


def _deduplicate_preserve_order(texts):
    """Remove duplicatas exatas preservando a ordem original."""
    unique = []
    seen = set()
    duplicates_removed = 0

    for text in texts:
        if text in seen:
            duplicates_removed += 1
            continue
        seen.add(text)
        unique.append(text)

    return unique, duplicates_removed


def _prepare_texts(texts, source_name, clean_payloads):
    """Normaliza, limpa e deduplica uma coleção de textos."""
    prepared = []
    removed_invalid = 0

    for text in texts:
        normalized = _normalize_text(text)
        if not normalized:
            removed_invalid += 1
            continue

        if clean_payloads:
            normalized = _clean_payload(normalized)
            if not normalized:
                removed_invalid += 1
                continue
            normalized = _normalize_text(normalized)
            if not normalized:
                removed_invalid += 1
                continue

        prepared.append(normalized)

    unique, duplicates_removed = _deduplicate_preserve_order(prepared)

    if removed_invalid > 0:
        print(f"  [Limpeza] {source_name}: {removed_invalid} entradas inválidas removidas")
    if duplicates_removed > 0:
        print(f"  [Deduplicação] {source_name}: {duplicates_removed} duplicatas exatas removidas")

    return unique


def _split_texts(texts, train_ratio, val_ratio, test_ratio, seed):
    """Divide textos em treino, validação e teste de forma determinística."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("As proporções de treino, validação e teste devem somar 1.0")

    if not texts:
        return [], [], []

    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    shuffled = [texts[i] for i in indices]
    total = len(shuffled)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    if total >= 1:
        train_end = max(train_end, 1)
    if total >= 3 and val_ratio > 0:
        val_end = max(val_end, train_end + 1)
        val_end = min(val_end, total - 1)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    return train, val, test


def _encode_texts(texts, char2idx, max_len):
    """Codifica uma coleção de textos em tensor."""
    if not texts:
        return None

    encoded = [encode_text(text, char2idx, max_len) for text in texts]
    return torch.tensor(encoded, dtype=torch.long)


_SQL_PATTERN = re.compile(
    r"(?:SELECT|INSERT|UPDATE|DELETE|DROP|UNION|WHERE|FROM|AND|OR|"
    r"ORDER\s+BY|GROUP\s+BY|HAVING|LIKE|BETWEEN|EXISTS|CREATE|ALTER|"
    r"TABLE|EXEC|CAST|CONCAT|SLEEP|BENCHMARK|WAITFOR|INFORMATION_SCHEMA|"
    r"LOAD_FILE|OUTFILE|CHAR\s*\(|CHR\s*\(|0x[0-9a-fA-F]|"
    r"['\";]\s*--|/\*|%27|%22|%3B|1\s*=\s*1|OR\s+\d)",
    re.IGNORECASE,
)

_STRONG_SQL = re.compile(
    r"(?:SELECT|UNION|DROP|INSERT|UPDATE|DELETE|EXEC|SLEEP|BENCHMARK|"
    r"WAITFOR|CONCAT|CHAR\s*\(|CHR\s*\(|0x[0-9a-fA-F]{4}|"
    r"INFORMATION_SCHEMA|LOAD_FILE|utl_inaddr|dbms_pipe|"
    r"'\s*(?:OR|AND)\s*'|--\s*$|/\*.*\*/|%27|%22|%3B)",
    re.IGNORECASE,
)

_ENGLISH_PROSE = re.compile(
    r"\b(?:movie|film|actor|actress|director|scene|plot|comedy|horror|"
    r"watch(?:ing|ed)?|story|character|audience|episode|season|"
    r"beautiful|terrible|wonderful|amazing|boring|funny|stupid|"
    r"recommend|performance|excellent|worst|best|laugh|hilarious|"
    r"romantic|thriller|cinema|screenplay|dialogue|viewer|series)\b",
    re.IGNORECASE,
)


def _clean_payload(text):
    """Limpa um payload SQLi removendo prosa em inglês após comentários SQL."""
    if not text or len(text) < 3:
        return None

    # Se tem muita prosa e NENHUM SQL forte, descartar
    prose_count = len(_ENGLISH_PROSE.findall(text))
    if prose_count >= 3 and not _STRONG_SQL.search(text):
        return None

    # Se não contém nenhum padrão SQL, descartar
    if not _SQL_PATTERN.search(text):
        return None

    # Truncar texto de prosa após indicadores de comentário SQL (-- ou /*)
    cleaned = text
    for marker in ["--", "/*"]:
        pos = cleaned.find(marker)
        if pos > 0:
            after = cleaned[pos + len(marker):]
            if _ENGLISH_PROSE.search(after):
                cleaned = cleaned[:pos + len(marker)].rstrip()

    # Verificação final: se ainda contém muita prosa vs SQL, descartar
    words = cleaned.split()
    if len(words) > 5:
        pc = len(_ENGLISH_PROSE.findall(cleaned))
        if pc >= 3 and not _STRONG_SQL.search(cleaned):
            return None

    return cleaned if len(cleaned) >= 3 else None


def _load_csv_query_label(path, max_samples=None, encoding="utf-8"):
    """Carrega CSV com colunas Query/Label (0=benigno, 1=malicioso)."""
    df = pd.read_csv(path, nrows=max_samples, encoding=encoding)

    # Normalizar nomes de colunas
    col_text = "Query" if "Query" in df.columns else "Sentence"
    col_label = "Label"

    df = df.dropna(subset=[col_text, col_label])
    df[col_label] = pd.to_numeric(df[col_label], errors="coerce")
    df = df.dropna(subset=[col_label])

    source_name = os.path.basename(path)
    raw_malicious = df[df[col_label] == 1][col_text].astype(str).tolist()
    raw_benign = df[df[col_label] == 0][col_text].astype(str).tolist()

    malicious = _prepare_texts(raw_malicious, source_name, clean_payloads=True)
    benign = _prepare_texts(raw_benign, source_name, clean_payloads=False)
    return malicious, benign


def _load_http_params(path, max_samples=None):
    """Carrega payloads do HttpParamsDataset. attack_type=='sqli' são ataques."""
    df = pd.read_csv(path, nrows=max_samples)

    source_name = os.path.basename(path)
    malicious = _prepare_texts(
        df[df["attack_type"] == "sqli"]["payload"].dropna().tolist(),
        source_name,
        clean_payloads=True,
    )
    benign = _prepare_texts(
        df[df["attack_type"] == "norm"]["payload"].dropna().tolist(),
        source_name,
        clean_payloads=False,
    )

    return malicious, benign


def _load_sqli_extended(path, max_samples=None):
    """Carrega payloads do sqli-extended. Label=1 (com limpeza de reviews)."""
    df = pd.read_csv(path, nrows=max_samples)

    source_name = os.path.basename(path)
    raw_malicious = df[df["Label"] == 1]["Sentence"].dropna().tolist()
    raw_benign = df[df["Label"] == 0]["Sentence"].dropna().tolist()

    malicious = _prepare_texts(raw_malicious, source_name, clean_payloads=True)
    benign = _prepare_texts(raw_benign, source_name, clean_payloads=False)
    return malicious, benign


def _load_txt_payloads(datasets_dir):
    """Carrega todos os arquivos .txt como payloads maliciosos (1 por linha)."""
    payloads = []
    txt_files = [f for f in os.listdir(datasets_dir) if f.endswith(".txt")]

    for filename in sorted(txt_files):
        path = os.path.join(datasets_dir, filename)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            payloads.extend(lines)
        except OSError as exc:
            print(f"[Aviso] Falha ao carregar {filename}: {exc}")
            continue

    return _prepare_texts(payloads, ".txt payloads", clean_payloads=True)


def load_all_datasets(datasets_dir="datasets", max_samples_per_dataset=None):
    """Carrega todos os datasets e retorna payloads maliciosos e benignos."""
    all_malicious = []
    all_benign = []

    # CSVs com formato Query/Label (utf-8)
    csv_query_label = [
        "clean_sql_dataset.csv",
        "Modified_SQL_Dataset.csv",
    ]

    # CSVs com formato Sentence/Label (utf-8)
    csv_sentence_label = [
        "sqli-extended.csv",
    ]

    # CSVs com formato Sentence/Label (utf-16)
    csv_utf16 = [
        "sqli.csv",
        "sqliv2.csv",
    ]

    # SQLiV3 (Sentence/Label com NaN)
    csv_sqliv3 = ["SQLiV3.csv"]

    # HttpParams (formato especial)
    http_params = "HttpParamsDataset_payload_full.csv"

    # --- Carregar CSVs Query/Label ---
    for filename in csv_query_label:
        path = os.path.join(datasets_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            malicious, benign = _load_csv_query_label(path, max_samples_per_dataset)
            all_malicious.extend(malicious)
            all_benign.extend(benign)
            print(f"[Dataset] {filename}: {len(malicious)} maliciosos, {len(benign)} benignos")
        except Exception as e:
            print(f"[ERRO] {filename}: {e}")

    # --- Carregar sqli-extended (com limpeza) ---
    for filename in csv_sentence_label:
        path = os.path.join(datasets_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            malicious, benign = _load_sqli_extended(path, max_samples_per_dataset)
            all_malicious.extend(malicious)
            all_benign.extend(benign)
            print(f"[Dataset] {filename}: {len(malicious)} maliciosos, {len(benign)} benignos")
        except Exception as e:
            print(f"[ERRO] {filename}: {e}")

    # --- Carregar CSVs utf-16 ---
    for filename in csv_utf16:
        path = os.path.join(datasets_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            malicious, benign = _load_csv_query_label(path, max_samples_per_dataset, encoding="utf-16")
            all_malicious.extend(malicious)
            all_benign.extend(benign)
            print(f"[Dataset] {filename}: {len(malicious)} maliciosos, {len(benign)} benignos")
        except Exception as e:
            print(f"[ERRO] {filename}: {e}")

    # --- SQLiV3 (float labels, NaN) ---
    for filename in csv_sqliv3:
        path = os.path.join(datasets_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            malicious, benign = _load_csv_query_label(path, max_samples_per_dataset)
            all_malicious.extend(malicious)
            all_benign.extend(benign)
            print(f"[Dataset] {filename}: {len(malicious)} maliciosos, {len(benign)} benignos")
        except Exception as e:
            print(f"[ERRO] {filename}: {e}")

    # --- HttpParams ---
    path = os.path.join(datasets_dir, http_params)
    if os.path.exists(path):
        try:
            malicious, benign = _load_http_params(path, max_samples_per_dataset)
            all_malicious.extend(malicious)
            all_benign.extend(benign)
            print(f"[Dataset] {http_params}: {len(malicious)} maliciosos, {len(benign)} benignos")
        except Exception as e:
            print(f"[ERRO] {http_params}: {e}")

    # --- Payload .txt files ---
    txt_payloads = _load_txt_payloads(datasets_dir)
    if txt_payloads:
        all_malicious.extend(txt_payloads)
        print(f"[Dataset] .txt payloads: {len(txt_payloads)} payloads carregados")

    all_malicious, removed_malicious = _deduplicate_preserve_order(all_malicious)
    all_benign, removed_benign = _deduplicate_preserve_order(all_benign)

    if removed_malicious > 0:
        print(f"[Deduplicação Global] maliciosos: {removed_malicious} duplicatas removidas")
    if removed_benign > 0:
        print(f"[Deduplicação Global] benignos: {removed_benign} duplicatas removidas")

    print(f"\n[Total] {len(all_malicious)} maliciosos, {len(all_benign)} benignos")
    return all_malicious, all_benign


def prepare_training_data(max_len=256, max_vocab_size=256,
                          datasets_dir="datasets", max_samples_per_dataset=None):
    """Pipeline completo de pré-processamento para o treinamento da GAN."""
    malicious, benign = load_all_datasets(datasets_dir, max_samples_per_dataset)

    if not malicious:
        raise ValueError("Nenhum payload malicioso encontrado nos datasets.")

    # Construir vocabulário usando payloads maliciosos (foco na geração de ataques)
    char2idx, idx2char = build_vocab(malicious, max_vocab_size)

    # Codificar payloads maliciosos
    encoded = [encode_text(text, char2idx, max_len) for text in malicious]
    tensor_data = torch.tensor(encoded, dtype=torch.long)

    # Codificar payloads benignos (para o discriminador)
    encoded_benign = [encode_text(text, char2idx, max_len) for text in benign]
    tensor_benign = torch.tensor(encoded_benign, dtype=torch.long) if encoded_benign else None

    vocab = {
        "char2idx": char2idx,
        "idx2char": idx2char,
        "vocab_size": len(char2idx),
        "max_len": max_len,
    }

    print(f"[Vocab] {vocab['vocab_size']} tokens | max_len={max_len}")
    print(f"[Dados] {tensor_data.shape[0]} sequências maliciosas codificadas")

    return tensor_data, tensor_benign, vocab


def prepare_training_splits(max_len=256, max_vocab_size=256,
                            datasets_dir="datasets", max_samples_per_dataset=None,
                            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                            split_seed=42):
    """Prepara splits determinísticos de treino, validação e teste."""
    malicious, benign = load_all_datasets(datasets_dir, max_samples_per_dataset)

    if not malicious:
        raise ValueError("Nenhum payload malicioso encontrado nos datasets.")

    train_mal, val_mal, test_mal = _split_texts(
        malicious, train_ratio, val_ratio, test_ratio, split_seed
    )
    train_ben, val_ben, test_ben = _split_texts(
        benign, train_ratio, val_ratio, test_ratio, split_seed + 1
    )

    vocab_texts = train_mal + train_ben if train_ben else train_mal
    char2idx, idx2char = build_vocab(vocab_texts, max_vocab_size)

    splits = {
        "train": {
            "malicious": _encode_texts(train_mal, char2idx, max_len),
            "benign": _encode_texts(train_ben, char2idx, max_len),
        },
        "val": {
            "malicious": _encode_texts(val_mal, char2idx, max_len),
            "benign": _encode_texts(val_ben, char2idx, max_len),
        },
        "test": {
            "malicious": _encode_texts(test_mal, char2idx, max_len),
            "benign": _encode_texts(test_ben, char2idx, max_len),
        },
    }

    vocab = {
        "char2idx": char2idx,
        "idx2char": idx2char,
        "vocab_size": len(char2idx),
        "max_len": max_len,
    }

    stats = {
        "train": {"malicious": len(train_mal), "benign": len(train_ben)},
        "val": {"malicious": len(val_mal), "benign": len(val_ben)},
        "test": {"malicious": len(test_mal), "benign": len(test_ben)},
    }

    print(f"[Split] treino={stats['train']} | validação={stats['val']} | teste={stats['test']}")
    print(f"[Vocab] {vocab['vocab_size']} tokens | max_len={max_len}")

    return splits, vocab, stats


def save_vocab(vocab, path):
    """Salva o vocabulário em JSON."""
    serializable = {
        "char2idx": vocab["char2idx"],
        "idx2char": {str(k): v for k, v in vocab["idx2char"].items()},
        "vocab_size": vocab["vocab_size"],
        "max_len": vocab["max_len"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def load_vocab(path):
    """Carrega o vocabulário de um arquivo JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["idx2char"] = {int(k): v for k, v in data["idx2char"].items()}
    return data
