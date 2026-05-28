import yaml
import json
import os
import requests


def load_swagger(file_path):
    # Suporta URLs remotas
    if file_path.startswith("http://") or file_path.startswith("https://"):
        resp = requests.get(file_path, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "json" in content_type or file_path.endswith(".json"):
            return resp.json()
        return yaml.safe_load(resp.text)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo Swagger não encontrado: {file_path}")

    with open(file_path, "r") as f:
        if file_path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def get_base_url(swagger):
    host = swagger.get("host", "localhost")
    base_path = swagger.get("basePath", "")
    schemes = swagger.get("schemes", ["http"])
    scheme = schemes[0] if schemes else "http"
    return f"{scheme}://{host}{base_path}"


def _resolve_ref(swagger, ref):
    parts = ref.lstrip("#/").split("/")
    node = swagger
    for part in parts:
        node = node.get(part, {})
    return node


def _extract_body_fields(swagger, parameters):
    fields = []
    for param in parameters:
        if param.get("in") == "body":
            schema = param.get("schema", {})

            if "$ref" in schema:
                schema = _resolve_ref(swagger, schema["$ref"])

            properties = schema.get("properties", {})
            for field_name, field_info in properties.items():
                fields.append({
                    "name": field_name,
                    "type": field_info.get("type", "string"),
                    "in": "body",
                    "required": field_name in schema.get("required", []),
                })
    return fields


def extract_endpoints(swagger):
    endpoints = []
    paths = swagger.get("paths", {})

    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue

        for method, details in methods.items():
            if method.lower() in ("parameters", "x-swagger-router-controller"):
                continue

            if not isinstance(details, dict):
                continue

            params = []
            raw_params = details.get("parameters", [])

            # Parâmetros query, path, header, formData
            for param in raw_params:
                if param.get("in") in ("query", "path", "header", "formData"):
                    params.append({
                        "name": param.get("name"),
                        "type": param.get("type", "string"),
                        "in": param.get("in"),
                        "required": param.get("required", False),
                    })

            # Parâmetros do body (extrair campos do schema)
            body_fields = _extract_body_fields(swagger, raw_params)
            params.extend(body_fields)

            endpoints.append({
                "path": path,
                "method": method.upper(),
                "params": params,
                "consumes": details.get("consumes", swagger.get("consumes", ["application/json"])),
                "produces": details.get("produces", swagger.get("produces", ["application/json"])),
                "operation_id": details.get("operationId", ""),
                "summary": details.get("summary", ""),
            })

    return endpoints


def print_endpoints_summary(endpoints):
    print(f"\n[Swagger] {len(endpoints)} endpoints mapeados:")
    for ep in endpoints:
        param_names = [p["name"] for p in ep["params"]]
        params_str = ", ".join(param_names) if param_names else "(sem parâmetros)"
        print(f"  {ep['method']:6s} {ep['path']} → {params_str}")
