import os
from scanner.gan.generate import (
    load_trained_generator,
    generate_payloads,
    generate_payloads_for_endpoint,
)


def _load_generator(model_path):
    """Carrega o gerador treinado."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo GAN não encontrado em '{model_path}'. "
            "Execute o treinamento primeiro: python main.py train"
        )
    return load_trained_generator(model_path)


def load_payloads(model_path="models/gan_sqli.pt", num_gan_payloads=200, temperature=0.7):
    """Carrega payloads genéricos (sem contexto de endpoint)."""
    print("Carregando modelo GAN treinado...")
    generator, vocab, config = _load_generator(model_path)

    gan_payloads = generate_payloads(
        generator, vocab, config,
        num_payloads=num_gan_payloads,
        temperature=temperature,
    )

    print(f"{len(gan_payloads)} payloads gerados pela GAN")
    return gan_payloads


def load_payloads_for_endpoint(endpoint, model_path="models/gan_sqli.pt",
                                num_payloads=50, temperature=0.7):
    """
    Gera payloads contextualizados com base no endpoint do Swagger.

    A GAN recebe as informações do endpoint (parâmetros, tipos, localização)
    e gera ataques direcionados para aquele endpoint específico.
    """
    generator, vocab, config = _load_generator(model_path)

    payloads = generate_payloads_for_endpoint(
        generator, vocab, config,
        endpoint=endpoint,
        num_payloads=num_payloads,
        base_temperature=temperature,
    )

    return payloads
