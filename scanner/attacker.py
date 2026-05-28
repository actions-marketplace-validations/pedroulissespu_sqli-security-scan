import time
import requests


REQUEST_TIMEOUT = 10


def _build_path_with_injection(path, params, payload):
    injected_path = path
    for param in params:
        if param["in"] == "path":
            placeholder = "{" + param["name"] + "}"
            # Usa um valor base realista antes do payload para que o
            # URL router reconheça o segmento e repasse à aplicação.
            param_type = param.get("type", "string")
            if param_type in ("integer", "number"):
                base_value = "1"
            else:
                base_value = "test"
            injected_path = injected_path.replace(placeholder, base_value + str(payload))
    return injected_path


def _build_query_params(params, payload):
    query = {}
    for param in params:
        if param["in"] == "query":
            query[param["name"]] = payload
    return query


def _build_body(params, payload):
    body = {}
    for param in params:
        if param["in"] in ("body", "formData"):
            body[param["name"]] = payload
    return body


def _build_headers(params, payload):
    headers = {}
    for param in params:
        if param["in"] == "header":
            headers[param["name"]] = str(payload)
    return headers


def attack_endpoint(base_url, endpoint, payloads, auth_token=None):
    results = []
    method = endpoint["method"].upper()
    params = endpoint.get("params", [])

    # Separar por tipo
    has_query = any(p["in"] == "query" for p in params)
    has_body = any(p["in"] in ("body", "formData") for p in params)
    has_path = any(p["in"] == "path" for p in params)
    has_header = any(p["in"] == "header" for p in params)

    # Pula caso não tenha parâmetros injetáveis
    if not params:
        return results

    for payload in payloads:
        # Fazer as URL com path params injetados
        path = endpoint["path"]
        if has_path:
            path = _build_path_with_injection(path, params, payload)

        url = base_url.rstrip("/") + path

        # Construir parâmetros por localização
        query_params = _build_query_params(params, payload) if has_query else None
        body_data = _build_body(params, payload) if has_body else None
        custom_headers = _build_headers(params, payload) if has_header else {}

        if auth_token:
            custom_headers["Authorization"] = auth_token

        # Determinar content-type
        consumes = endpoint.get("consumes", ["application/json"])
        use_json = "application/json" in consumes

        try:
            start_time = time.time()

            kwargs = {
                "url": url,
                "timeout": REQUEST_TIMEOUT,
                "headers": custom_headers,
            }

            if query_params:
                kwargs["params"] = query_params

            if body_data:
                if use_json:
                    kwargs["json"] = body_data
                else:
                    kwargs["data"] = body_data

            response = requests.request(method, **kwargs)
            elapsed = time.time() - start_time

            results.append({
                "payload": payload,
                "method": method,
                "url": url,
                "status": response.status_code,
                "response": response.text[:2000],
                "response_time": round(elapsed, 4),
                "content_length": len(response.text),
            })

        except requests.Timeout:
            results.append({
                "payload": payload,
                "method": method,
                "url": url,
                "error": "timeout",
                "response_time": REQUEST_TIMEOUT,
            })
        except requests.ConnectionError:
            results.append({
                "payload": payload,
                "method": method,
                "url": url,
                "error": "connection_refused",
            })
        except Exception as e:
            results.append({
                "payload": payload,
                "method": method,
                "url": url,
                "error": str(e),
            })

    return results
