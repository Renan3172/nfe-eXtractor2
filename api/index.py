import base64
import html
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

# Modelos Flash atuais testados em ordem de velocidade e disponibilidade
MODELOS_GEMINI_CANDIDATOS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

# Cache leve em memória para reutilizar direto o primeiro modelo que responder com sucesso
_CACHE_GEMINI: Dict[str, Tuple[str, str]] = {}

# Endpoints públicos de consulta pela chave de 44 dígitos
CONSULTA_DANFE_WEB_URL = "https://consultadanfe.com/CDanfe/chave"
CONSULTA_DANFE_API_URL = "https://consultadanfe.com/api/v1/consulta"
CONSULTA_DACTE_WEB_URL = "https://consultadacte.com/CDacte/chave"

PROMPT_OCR_FISCAL = """
Você é um especialista em OCR de documentos fiscais brasileiros (DANFE de NF-e/NFC-e e DACTE de CT-e/CT-e OS).
Analise a imagem (mesmo que esteja inclinada, rotacionada, borrada ou com baixa iluminação) e extraia EXCLUSIVAMENTE a Chave de Acesso de 44 dígitos numéricos.

Onde procurar:
1. Localize o bloco "CHAVE DE ACESSO" no DANFE ou no DACTE, geralmente logo abaixo do código de barras principal.
2. Os 44 dígitos costumam aparecer em 11 grupos de 4 números separados por espaços.

Regras obrigatórias:
- A chave possui EXATAMENTE 44 dígitos numéricos (0 a 9).
- Os dígitos 21 e 22 indicam o modelo do documento: "55" (NF-e), "65" (NFC-e), "57" (CT-e) ou "67" (CT-e OS).
- Tenha atenção redobrada para não confundir dígitos borrados (0 e 8, 1 e 7, 5 e 6, 3 e 8).
- Remova todos os espaços, pontos, traços ou letras.

Formato de saída:
Retorne SOMENTE os 44 dígitos contínuos, sem texto explicativo, sem markdown e sem quebras de linha.
Se não for possível ler os 44 dígitos com segurança, retorne apenas: ERRO_LEITURA
""".strip()

MAPA_MODELOS = {
    "55": "NF-e",
    "65": "NFC-e",
    "57": "CT-e",
    "67": "CT-e OS",
}

MAPA_MOD_FRETE = {
    "0": "REMETENTE",
    "1": "DESTINATÁRIO",
    "2": "TERCEIROS",
    "3": "REMETENTE",
    "4": "DESTINATÁRIO",
    "9": "SEM FRETE",
}


def validar_chave_acesso(chave: str) -> bool:
    """Valida formato de 44 dígitos, modelo fiscal suportado e Dígito Verificador (Módulo 11)."""
    if not chave or len(chave) != 44 or not chave.isdigit():
        return False

    if chave[20:22] not in MAPA_MODELOS:
        return False

    peso = 2
    soma = 0
    for digito in reversed(chave[:43]):
        soma += int(digito) * peso
        peso = 2 if peso == 9 else peso + 1

    resto = soma % 11
    dv_esperado = 0 if resto in (0, 1) else 11 - resto
    return dv_esperado == int(chave[43])


def limpar_digitos(valor: Optional[str]) -> str:
    return re.sub(r"\D", "", valor or "")


def formatar_documento(digitos: str) -> str:
    if len(digitos) == 14:
        return f"{digitos[:2]}.{digitos[2:5]}.{digitos[5:8]}/{digitos[8:12]}-{digitos[12:]}"
    if len(digitos) == 11:
        return f"{digitos[:3]}.{digitos[3:6]}.{digitos[6:9]}-{digitos[9:]}"
    return digitos


def verificar_janela_data_chave(chave: str) -> Optional[str]:
    """Verifica se a emissão (AAMM nos dígitos 3..6) está fora da janela gratuita do ConsultaDanfe."""
    try:
        ano = 2000 + int(chave[2:4])
        mes = int(chave[4:6])
        if not (1 <= mes <= 12):
            return None
    except ValueError:
        return None

    agora_br = datetime.now(timezone(timedelta(hours=-3)))
    mes_atual = (agora_br.year, agora_br.month)

    if agora_br.day < 15:
        if agora_br.month == 1:
            mes_minimo = (agora_br.year - 1, 12)
        else:
            mes_minimo = (agora_br.year, agora_br.month - 1)
    else:
        mes_minimo = mes_atual

    if (ano, mes) < mes_minimo:
        return (
            f"Documento emitido em {mes:02d}/{ano}. A consulta pública gratuita do ConsultaDanfe "
            "aceita apenas notas do mês corrente (ou do mês anterior até o dia 14)."
        )
    return None


def _extrair_mensagem_erro_google(resp: requests.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
    except ValueError:
        pass
    return f"HTTP {resp.status_code}"


def _descobrir_modelos_aistudio(api_key: str) -> List[str]:
    """Consulta modelos Flash multimodais ativos na conta caso todos os fixos falhem."""
    try:
        resp = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": api_key},
            timeout=10,
        )
        if not resp.ok:
            return []
        modelos = resp.json().get("models", [])
    except Exception:
        return []

    termos_ignorar = (
        "embedding",
        "tts",
        "live",
        "image",
        "imagen",
        "audio",
        "transcribe",
        "veo",
        "omni",
        "cyber",
    )

    preferidos: List[str] = []
    alternativos: List[str] = []

    for item in modelos:
        nome_completo = str(item.get("name", ""))
        metodos = item.get("supportedGenerationMethods", [])
        if "generateContent" not in metodos:
            continue
        slug = nome_completo.replace("models/", "").strip()
        if not slug or any(t in slug for t in termos_ignorar):
            continue
        if "gemini" in slug and "flash" in slug:
            preferidos.append(slug)
        elif "gemini" in slug:
            alternativos.append(slug)

    return preferidos + alternativos


def _montar_tentativas_gemini(api_key: str) -> List[Tuple[str, str]]:
    """Define a ordem de (provedor, modelo) evitando rotas incompatíveis com o prefixo da chave."""
    chave_limpa = api_key.strip()
    eh_vertex = chave_limpa.startswith("AQ.")
    eh_aistudio = chave_limpa.startswith("AIza")
    cache_key = "vertex" if eh_vertex else "aistudio"

    tentativas: List[Tuple[str, str]] = []
    if cache_key in _CACHE_GEMINI:
        tentativas.append(_CACHE_GEMINI[cache_key])

    if eh_vertex:
        ordem_provedores = ["vertex", "aistudio"]
    elif eh_aistudio:
        ordem_provedores = ["aistudio"]
    else:
        ordem_provedores = ["aistudio", "vertex"]

    for provedor in ordem_provedores:
        for modelo in MODELOS_GEMINI_CANDIDATOS:
            par = (provedor, modelo)
            if par not in tentativas:
                tentativas.append(par)

    return tentativas


def _chamar_endpoint_gemini(
    provedor: str, modelo: str, api_key: str, payload: Dict[str, Any]
) -> requests.Response:
    if provedor == "vertex":
        url = (
            f"https://aiplatform.googleapis.com/v1/publishers/google/"
            f"models/{modelo}:generateContent"
        )
        return requests.post(
            url,
            params={"key": api_key},
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{modelo}:generateContent"
    )
    return requests.post(
        url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        timeout=15,
    )


def extrair_chave_via_gemini(
    imagem_bytes: bytes, mime_type: str, api_key: str
) -> Tuple[Optional[str], Optional[str]]:
    """Envia a imagem para o Gemini com fallback automático em 404, 429 e 503."""
    chave_limpa = api_key.strip()
    imagem_b64 = base64.b64encode(imagem_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": PROMPT_OCR_FISCAL},
                    {
                        "inline_data": {
                            "mime_type": mime_type or "image/jpeg",
                            "data": imagem_b64,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
        },
    }

    tentativas = _montar_tentativas_gemini(chave_limpa)
    resp_sucesso: Optional[requests.Response] = None
    ultimo_erro = "Erro desconhecido na API do Gemini."
    cache_key = "vertex" if chave_limpa.startswith("AQ.") else "aistudio"
    teve_indisponibilidade = False
    teve_cota = False
    testados: List[Tuple[str, str]] = []

    for provedor, modelo in tentativas:
        testados.append((provedor, modelo))
        try:
            resp = _chamar_endpoint_gemini(provedor, modelo, chave_limpa, payload)
        except requests.Timeout:
            teve_indisponibilidade = True
            ultimo_erro = f"Timeout ({provedor}/{modelo})"
            if _CACHE_GEMINI.get(cache_key) == (provedor, modelo):
                _CACHE_GEMINI.pop(cache_key, None)
            continue
        except requests.RequestException:
            return None, "Falha de conexão ao comunicar com a API do Gemini."

        if resp.ok:
            resp_sucesso = resp
            _CACHE_GEMINI[cache_key] = (provedor, modelo)
            break

        if _CACHE_GEMINI.get(cache_key) == (provedor, modelo):
            _CACHE_GEMINI.pop(cache_key, None)

        detalhe = _extrair_mensagem_erro_google(resp)
        ultimo_erro = f"HTTP {resp.status_code} ({provedor}/{modelo}): {detalhe}"

        if resp.status_code in (500, 502, 503, 504):
            teve_indisponibilidade = True
            continue

        if resp.status_code == 429:
            teve_cota = True
            continue

        if resp.status_code in (400, 401, 403, 404):
            continue

        return None, f"Erro retornado pela API do Gemini ({ultimo_erro})."

    if resp_sucesso is None and not chave_limpa.startswith("AQ."):
        for modelo_extra in _descobrir_modelos_aistudio(chave_limpa):
            par = ("aistudio", modelo_extra)
            if par in testados:
                continue
            testados.append(par)
            try:
                resp = _chamar_endpoint_gemini("aistudio", modelo_extra, chave_limpa, payload)
                if resp.ok:
                    resp_sucesso = resp
                    _CACHE_GEMINI[cache_key] = par
                    break
                if resp.status_code in (500, 502, 503, 504):
                    teve_indisponibilidade = True
                ultimo_erro = (
                    f"HTTP {resp.status_code} (aistudio/{modelo_extra}): "
                    f"{_extrair_mensagem_erro_google(resp)}"
                )
            except requests.RequestException:
                continue
            if len(testados) >= len(tentativas) + 2:
                break

    if resp_sucesso is None and teve_indisponibilidade:
        time.sleep(1.2)
        provedor_retry = "vertex" if chave_limpa.startswith("AQ.") else "aistudio"
        for modelo_retry in MODELOS_GEMINI_CANDIDATOS[:2]:
            try:
                resp = _chamar_endpoint_gemini(provedor_retry, modelo_retry, chave_limpa, payload)
                if resp.ok:
                    resp_sucesso = resp
                    _CACHE_GEMINI[cache_key] = (provedor_retry, modelo_retry)
                    break
                ultimo_erro = (
                    f"HTTP {resp.status_code} ({provedor_retry}/{modelo_retry}): "
                    f"{_extrair_mensagem_erro_google(resp)}"
                )
            except requests.RequestException:
                continue

    if resp_sucesso is None:
        if teve_indisponibilidade:
            return (
                None,
                "Os servidores da API do Gemini estão com alta demanda momentânea (HTTP 503). "
                "Aguarde alguns segundos e clique em Enviar novamente.",
            )
        if teve_cota:
            return (
                None,
                "Limite de requisições da sua chave Gemini atingido. Aguarde alguns segundos.",
            )
        return (
            None,
            f"Erro retornado pela API do Gemini ({ultimo_erro}). "
            "Verifique se a chave está correta em Configurar API.",
        )

    try:
        dados = resp_sucesso.json()
        texto = dados["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, ValueError, TypeError):
        return None, "Não foi possível interpretar a resposta da IA para esta imagem."

    if "ERRO_LEITURA" in texto.upper():
        return (
            None,
            "Não foi possível identificar a chave de acesso de 44 dígitos. "
            "Tente enviar uma foto mais próxima, reta e sob boa luz.",
        )

    chave = limpar_digitos(texto)
    if not validar_chave_acesso(chave):
        return (
            None,
            "A chave lida pela IA não passou na validação matemática (Módulo 11) ou "
            "não corresponde a NF-e/CT-e. Tente uma foto mais nítida.",
        )

    return chave, None


def _eh_xml_fiscal(conteudo: bytes) -> bool:
    inicio = conteudo.lstrip()[:600]
    return any(
        marcador in inicio or marcador in conteudo[:4000]
        for marcador in (b"<nfeProc", b"<NFe", b"<infNFe", b"<cteProc", b"<CTe", b"<infCte")
    )


def _tentar_decodificar_xml(valor: str, session: requests.Session, headers: Dict[str, str]) -> Optional[bytes]:
    texto = valor.strip()
    if not texto:
        return None

    # XML cru ou escapado em HTML (&lt;nfeProc...)
    if texto.startswith("<") or "&lt;" in texto[:40]:
        desescapado = html.unescape(texto)
        raw = desescapado.encode("utf-8", errors="ignore")
        if _eh_xml_fiscal(raw):
            return raw

    # URL direta para XML
    if texto.startswith("http://") or texto.startswith("https://"):
        try:
            r = session.get(texto, headers=headers, timeout=12)
            if r.ok and _eh_xml_fiscal(r.content):
                return r.content
        except requests.RequestException:
            return None

    # Base64 (com ou sem prefixo data:)
    candidato_b64 = texto.split(",", 1)[1] if texto.startswith("data:") and "," in texto else texto
    if len(candidato_b64) >= 60:
        try:
            decoded = base64.b64decode(candidato_b64, validate=False)
            if _eh_xml_fiscal(decoded):
                return decoded
        except Exception:
            pass

    return None


def _tentar_decodificar_pdf(valor: str, session: requests.Session, headers: Dict[str, str]) -> Optional[bytes]:
    texto = valor.strip()
    if not texto:
        return None

    if texto.startswith("http://") or texto.startswith("https://"):
        try:
            r = session.get(texto, headers=headers, timeout=12)
            if r.ok and r.content.lstrip().startswith(b"%PDF"):
                return r.content
        except requests.RequestException:
            return None

    candidato_b64 = texto.split(",", 1)[1] if texto.startswith("data:") and "," in texto else texto
    if candidato_b64.startswith("JVBERi") or len(candidato_b64) >= 200:
        try:
            decoded = base64.b64decode(candidato_b64, validate=False)
            if decoded.lstrip().startswith(b"%PDF"):
                return decoded
        except Exception:
            pass

    return None


def _extrair_de_json_recursivo(
    obj: Any, session: requests.Session, headers: Dict[str, str]
) -> Tuple[Optional[bytes], Optional[bytes], Optional[str]]:
    """Percorre qualquer estrutura JSON para localizar XML, PDF e eventuais mensagens de erro."""
    xml_bytes: Optional[bytes] = None
    pdf_bytes: Optional[bytes] = None
    msg_erro: Optional[str] = None

    def _visitar(item: Any, chave_pai: str = "") -> None:
        nonlocal xml_bytes, pdf_bytes, msg_erro

        if isinstance(item, dict):
            for k, v in item.items():
                k_low = str(k).lower()
                if isinstance(v, str) and v.strip():
                    if not xml_bytes and (
                        "xml" in k_low
                        or v.lstrip().startswith("<")
                        or "&lt;" in v[:40]
                    ):
                        achado = _tentar_decodificar_xml(v, session, headers)
                        if achado:
                            xml_bytes = achado
                            continue

                    if not pdf_bytes and (
                        "pdf" in k_low
                        or "danfe" in k_low
                        or "dacte" in k_low
                        or v.startswith("JVBERi")
                        or v.startswith("data:application/pdf")
                    ):
                        achado_pdf = _tentar_decodificar_pdf(v, session, headers)
                        if achado_pdf:
                            pdf_bytes = achado_pdf
                            continue

                    if not msg_erro and k_low in ("message", "mensagem", "erro", "error", "detail", "msg", "motivo"):
                        msg_erro = v.strip()
                else:
                    _visitar(v, k_low)

        elif isinstance(item, list):
            for elem in item:
                _visitar(elem, chave_pai)

        elif isinstance(item, str) and item.strip():
            if not xml_bytes:
                achado = _tentar_decodificar_xml(item, session, headers)
                if achado:
                    xml_bytes = achado
            if not pdf_bytes:
                achado_pdf = _tentar_decodificar_pdf(item, session, headers)
                if achado_pdf:
                    pdf_bytes = achado_pdf

    _visitar(obj)
    return xml_bytes, pdf_bytes, msg_erro


def _analisar_resposta_consulta(
    resp: requests.Response, session: requests.Session, headers: Dict[str, str]
) -> Tuple[Optional[bytes], Optional[bytes], Optional[str]]:
    """Extrai (xml_bytes, pdf_bytes, diagnostico) de uma resposta HTTP."""
    content_type = resp.headers.get("Content-Type", "").lower()

    if resp.content.lstrip().startswith(b"%PDF"):
        return None, resp.content, f"HTTP {resp.status_code} retornou apenas PDF sem XML."

    if _eh_xml_fiscal(resp.content):
        return resp.content, None, None

    if "json" in content_type or resp.text.lstrip().startswith("{"):
        try:
            payload = resp.json()
        except ValueError:
            return None, None, f"HTTP {resp.status_code} com JSON inválido."

        xml_b, pdf_b, msg = _extrair_de_json_recursivo(payload, session, headers)
        if xml_b or pdf_b:
            return xml_b, pdf_b, msg

        if msg:
            return None, None, f"HTTP {resp.status_code}: {msg}"

        chaves_json = ", ".join(list(payload.keys())[:6]) if isinstance(payload, dict) else type(payload).__name__
        return None, None, f"HTTP {resp.status_code} sem XML no JSON (chaves: {chaves_json})."

    # Resposta HTML (bloqueio anti-bot, sessão ou rota alterada)
    if "html" in content_type or resp.text.lstrip().startswith("<"):
        if "cloudflare" in resp.text.lower() or "just a moment" in resp.text.lower():
            return None, None, f"HTTP {resp.status_code}: proteção anti-bot (Cloudflare) bloqueou a consulta."
        return None, None, f"HTTP {resp.status_code}: servidor retornou página HTML em vez de XML/JSON."

    return None, None, f"HTTP {resp.status_code} ({content_type or 'sem content-type'})."


def consultar_endpoint_publico(
    chave: str, modelo: str
) -> Tuple[Optional[bytes], Optional[bytes], Optional[str]]:
    """
    Consulta XML e PDF nos endpoints públicos disponíveis:
    1) ConsultaDanfe web (CDanfe/chave com sessão e cabeçalhos AJAX)
    2) ConsultaDanfe REST API pública (/api/v1/consulta)
    3) ConsultaDacte web (fallback específico para CT-e 57/67)
    """
    session = requests.Session()
    headers_base = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    xml_final: Optional[bytes] = None
    pdf_final: Optional[bytes] = None
    diagnosticos: List[str] = []

    # Tentativa 1: ConsultaDanfe rota web (CDanfe/chave) com cookies de sessão
    headers_cdanfe = {
        **headers_base,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Origin": "https://consultadanfe.com",
        "Referer": "https://consultadanfe.com/",
        "X-Requested-With": "XMLHttpRequest",
    }
    try:
        try:
            session.get("https://consultadanfe.com/", headers=headers_base, timeout=6)
        except requests.RequestException:
            pass

        resp1 = session.post(
            CONSULTA_DANFE_WEB_URL,
            data={"chave": chave},
            headers=headers_cdanfe,
            timeout=15,
        )
        xml1, pdf1, diag1 = _analisar_resposta_consulta(resp1, session, headers_cdanfe)
        if xml1:
            xml_final = xml1
        if pdf1:
            pdf_final = pdf1
        if diag1 and not xml1:
            diagnosticos.append(f"CDanfe: {diag1}")
    except requests.Timeout:
        diagnosticos.append("CDanfe: tempo limite excedido")
    except requests.RequestException:
        diagnosticos.append("CDanfe: falha de conexão")

    # Tentativa 2: ConsultaDanfe API REST pública v1 (busca XML/PDF se ainda faltar algum)
    if not xml_final or not pdf_final:
        headers_api = {
            **headers_base,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        try:
            resp2 = session.post(
                CONSULTA_DANFE_API_URL,
                json={"chave": chave},
                headers=headers_api,
                timeout=15,
            )
            xml2, pdf2, diag2 = _analisar_resposta_consulta(resp2, session, headers_api)
            if xml2 and not xml_final:
                xml_final = xml2
            if pdf2 and not pdf_final:
                pdf_final = pdf2
            if diag2 and not xml_final:
                diagnosticos.append(f"API v1: {diag2}")
        except requests.Timeout:
            if not xml_final:
                diagnosticos.append("API v1: tempo limite excedido")
        except requests.RequestException:
            if not xml_final:
                diagnosticos.append("API v1: falha de conexão")

    # Tentativa 3: Fallback específico para CT-e (57/67)
    if not xml_final and modelo in ("57", "67"):
        headers_dacte = {
            **headers_base,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Origin": "https://consultadacte.com",
            "Referer": "https://consultadacte.com/",
            "X-Requested-With": "XMLHttpRequest",
        }
        try:
            resp3 = session.post(
                CONSULTA_DACTE_WEB_URL,
                data={"chave": chave},
                headers=headers_dacte,
                timeout=15,
            )
            xml3, pdf3, diag3 = _analisar_resposta_consulta(resp3, session, headers_dacte)
            if xml3:
                xml_final = xml3
            if pdf3 and not pdf_final:
                pdf_final = pdf3
            if diag3 and not xml_final:
                diagnosticos.append(f"CDacte: {diag3}")
        except requests.Timeout:
            diagnosticos.append("CDacte: tempo limite excedido")
        except requests.RequestException:
            diagnosticos.append("CDacte: falha de conexão")

    if xml_final:
        return xml_final, pdf_final, None

    aviso_data = verificar_janela_data_chave(chave)
    resumo_diag = " | ".join(diagnosticos) if diagnosticos else "nenhum XML retornado."

    if aviso_data:
        return None, pdf_final, f"{aviso_data} Detalhe: {resumo_diag}"

    return (
        None,
        pdf_final,
        f"A chave foi lida, mas o XML não pôde ser obtido. Detalhe: {resumo_diag}",
    )


def _tag_local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _buscar_filho(elem: Optional[ET.Element], nome: str) -> Optional[ET.Element]:
    if elem is None:
        return None
    for filho in elem:
        if _tag_local(filho.tag) == nome:
            return filho
    return None


def _buscar_primeiro(root: ET.Element, nome: str) -> Optional[ET.Element]:
    for elem in root.iter():
        if _tag_local(elem.tag) == nome:
            return elem
    return None


def _texto_filho(elem: Optional[ET.Element], nome: str) -> Optional[str]:
    alvo = _buscar_filho(elem, nome)
    if alvo is not None and alvo.text and alvo.text.strip():
        return alvo.text.strip()
    return None


def _extrair_entidade(bloco: Optional[ET.Element]) -> Optional[Dict[str, str]]:
    if bloco is None:
        return None

    doc = limpar_digitos(_texto_filho(bloco, "CNPJ") or _texto_filho(bloco, "CPF"))
    razao = _texto_filho(bloco, "xNome")

    if not doc and not razao:
        return None

    return {
        "cnpj_limpo": doc,
        "cnpj_formatado": formatar_documento(doc) if doc else "NULO",
        "razao_social": razao or "NULO",
    }


def extrair_dados_xml(xml_bytes: bytes, modelo: str) -> Dict[str, Any]:
    """Lê o XML da NF-e/NFC-e ou CT-e/CT-e OS e retorna Remetente, Destinatário e Tomador."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return {
            "remetente": None,
            "destinatario": None,
            "tomador": {"modo": "TEXTO", "texto": "NULO"},
        }

    # NF-e (55) e NFC-e (65)
    if modelo in ("55", "65"):
        inf_nfe = _buscar_primeiro(root, "infNFe") or root
        remetente = _extrair_entidade(_buscar_filho(inf_nfe, "emit"))
        destinatario = _extrair_entidade(_buscar_filho(inf_nfe, "dest"))

        transp = _buscar_filho(inf_nfe, "transp")
        mod_frete = _texto_filho(transp, "modFrete")
        tomador_texto = MAPA_MOD_FRETE.get(mod_frete or "", "NULO")

        return {
            "remetente": remetente,
            "destinatario": destinatario,
            "tomador": {
                "modo": "TEXTO",
                "texto": tomador_texto,
            },
        }

    # CT-e (57) e CT-e OS (67)
    inf_cte = _buscar_primeiro(root, "infCte") or root
    rem_elem = _buscar_filho(inf_cte, "rem")
    dest_elem = _buscar_filho(inf_cte, "dest")
    exped_elem = _buscar_filho(inf_cte, "exped")
    receb_elem = _buscar_filho(inf_cte, "receb")

    remetente = _extrair_entidade(rem_elem)
    destinatario = _extrair_entidade(dest_elem)

    ide = _buscar_filho(inf_cte, "ide")
    tomador_entidade: Optional[Dict[str, str]] = None

    toma3 = _buscar_filho(ide, "toma3") or _buscar_filho(ide, "toma03")
    toma4 = _buscar_filho(ide, "toma4") or _buscar_filho(ide, "toma04")

    if toma4 is not None:
        tomador_entidade = _extrair_entidade(toma4)
    elif toma3 is not None:
        codigo_toma = _texto_filho(toma3, "toma")
        mapa_toma_cte = {
            "0": rem_elem,
            "1": exped_elem,
            "2": receb_elem,
            "3": dest_elem,
        }
        tomador_entidade = _extrair_entidade(mapa_toma_cte.get(codigo_toma or ""))
    else:
        toma_os = _buscar_filho(inf_cte, "toma")
        if toma_os is not None:
            tomador_entidade = _extrair_entidade(toma_os)

    if tomador_entidade:
        tomador_payload: Dict[str, Any] = {
            "modo": "ENTIDADE",
            **tomador_entidade,
        }
    else:
        tomador_payload = {
            "modo": "TEXTO",
            "texto": "NULO",
        }

    return {
        "remetente": remetente,
        "destinatario": destinatario,
        "tomador": tomador_payload,
    }


@app.post("/api/processar-nota")
async def processar_nota(
    imagem: UploadFile = File(...),
    gemini_api_key: str = Form(...),
):
    if not gemini_api_key or not gemini_api_key.strip():
        return JSONResponse(
            status_code=400,
            content={
                "erro_tipo": "LEITURA",
                "detail": "Configure sua chave de API do Gemini antes de enviar o documento.",
            },
        )

    imagem_bytes = await imagem.read()
    if not imagem_bytes:
        return JSONResponse(
            status_code=400,
            content={
                "erro_tipo": "LEITURA",
                "detail": "Nenhuma imagem válida foi recebida pelo servidor.",
            },
        )

    # Etapa 1: OCR Fiscal com Gemini + validação Módulo 11
    chave, erro_leitura = extrair_chave_via_gemini(
        imagem_bytes=imagem_bytes,
        mime_type=imagem.content_type or "image/jpeg",
        api_key=gemini_api_key,
    )

    if erro_leitura or not chave:
        return JSONResponse(
            status_code=422,
            content={
                "erro_tipo": "LEITURA",
                "detail": erro_leitura or "Não foi possível ler a chave de acesso.",
            },
        )

    modelo = chave[20:22]
    tipo_documento = MAPA_MODELOS.get(modelo, "NF-e")

    # Etapa 2: Consulta no endpoint público correspondente
    xml_bytes, pdf_bytes, erro_consulta = consultar_endpoint_publico(chave, modelo)

    if erro_consulta or not xml_bytes:
        return JSONResponse(
            status_code=200,
            content={
                "erro_tipo": "CONSULTA",
                "detail": erro_consulta
                or "A chave foi lida, mas não foi possível consultar os dados do documento.",
                "chave_acesso": chave,
                "tipo_documento": tipo_documento,
                "remetente": None,
                "destinatario": None,
                "tomador": None,
                "xml_base64": None,
                "pdf_base64": base64.b64encode(pdf_bytes).decode("utf-8") if pdf_bytes else None,
            },
        )

    # Etapa 3: Extração dos campos diretamente do XML
    dados_fiscais = extrair_dados_xml(xml_bytes, modelo)

    return JSONResponse(
        status_code=200,
        content={
            "erro_tipo": None,
            "detail": None,
            "chave_acesso": chave,
            "tipo_documento": tipo_documento,
            "remetente": dados_fiscais["remetente"],
            "destinatario": dados_fiscais["destinatario"],
            "tomador": dados_fiscais["tomador"],
            "xml_base64": base64.b64encode(xml_bytes).decode("utf-8"),
            "pdf_base64": base64.b64encode(pdf_bytes).decode("utf-8") if pdf_bytes else None,
        },
    )
