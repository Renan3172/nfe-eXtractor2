import base64
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Tuple

import requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

# Endpoints públicos de consulta pela chave de 44 dígitos
CONSULTA_DANFE_URL = "https://consultadanfe.com/CDanfe/chave"
CONSULTA_DACTE_URL = "https://consultadacte.com/CDacte/chave"

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


def extrair_chave_via_gemini(
    imagem_bytes: bytes, mime_type: str, api_key: str
) -> Tuple[Optional[str], Optional[str]]:
    """Envia a imagem para o Gemini e retorna (chave_44_digitos, mensagem_erro)."""
    imagem_b64 = base64.b64encode(imagem_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": PROMPT_OCR_FISCAL},
                    {
                        "inline_data": {
                            "mime_type": mime_type or "image/jpeg",
                            "data": imagem_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key.strip(),
    }

    try:
        resp = requests.post(GEMINI_ENDPOINT, json=payload, headers=headers, timeout=25)
    except requests.Timeout:
        return None, "Tempo limite excedido ao conectar com a API do Gemini. Tente novamente."
    except requests.RequestException:
        return None, "Falha de conexão ao comunicar com a API do Gemini."

    if resp.status_code in (400, 401, 403):
        return None, "Chave de API do Gemini inválida ou sem permissão. Verifique em Configurar API."
    if resp.status_code == 429:
        return None, "Limite de requisições da sua chave Gemini atingido. Aguarde alguns segundos."
    if not resp.ok:
        return None, f"Erro retornado pela API do Gemini (HTTP {resp.status_code})."

    try:
        dados = resp.json()
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


def consultar_endpoint_publico(
    chave: str, modelo: str
) -> Tuple[Optional[bytes], Optional[bytes], Optional[str]]:
    """
    Consulta o XML e o PDF via endpoint público correspondente:
    - Modelos 55 / 65 -> ConsultaDanfe
    - Modelos 57 / 67 -> ConsultaDacte
    Retorna (xml_bytes, pdf_bytes, mensagem_erro).
    """
    url = CONSULTA_DACTE_URL if modelo in ("57", "67") else CONSULTA_DANFE_URL
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
    }

    try:
        resp = requests.post(url, data={"chave": chave}, headers=headers, timeout=25)
    except requests.Timeout:
        return None, None, "Tempo limite excedido ao consultar o serviço fiscal público."
    except requests.RequestException:
        return None, None, "Não foi possível conectar ao serviço público de consulta fiscal."

    if resp.status_code in (403, 429):
        return (
            None,
            None,
            "O serviço de consulta limitou temporariamente as requisições. Aguarde e tente novamente.",
        )
    if not resp.ok:
        return (
            None,
            None,
            f"A chave foi lida, mas o serviço de consulta retornou erro (HTTP {resp.status_code}).",
        )

    xml_bytes: Optional[bytes] = None
    pdf_bytes: Optional[bytes] = None

    content_type = resp.headers.get("Content-Type", "").lower()

    if "application/json" in content_type:
        try:
            payload = resp.json()
        except ValueError:
            payload = {}

        if isinstance(payload, dict):
            # Compatível com retornos em base64, XML cru ou URL direta
            raw_xml = payload.get("xml") or payload.get("codigo_xml") or payload.get("xml_base64")
            raw_pdf = payload.get("pdf") or payload.get("pdf_base64") or payload.get("danfe") or payload.get("dacte")

            if isinstance(raw_xml, str) and raw_xml.strip():
                if raw_xml.lstrip().startswith("<"):
                    xml_bytes = raw_xml.encode("utf-8")
                elif raw_xml.startswith("http"):
                    r_xml = requests.get(raw_xml, headers=headers, timeout=15)
                    if r_xml.ok and r_xml.content.lstrip().startswith(b"<"):
                        xml_bytes = r_xml.content
                else:
                    try:
                        decoded = base64.b64decode(raw_xml)
                        if decoded.lstrip().startswith(b"<"):
                            xml_bytes = decoded
                    except Exception:
                        pass

            if isinstance(raw_pdf, str) and raw_pdf.strip():
                if raw_pdf.startswith("http"):
                    r_pdf = requests.get(raw_pdf, headers=headers, timeout=15)
                    if r_pdf.ok:
                        pdf_bytes = r_pdf.content
                else:
                    try:
                        pdf_bytes = base64.b64decode(raw_pdf)
                    except Exception:
                        pass

            if not xml_bytes and payload.get("status") == "error":
                msg = payload.get("message") or "Documento não encontrado ou não autorizado."
                return None, None, str(msg)

    elif "xml" in content_type or resp.content.lstrip().startswith(b"<"):
        # Se a resposta vier diretamente em XML
        if b"<nfeProc" in resp.content or b"<NFe" in resp.content or b"<cteProc" in resp.content or b"<CTe" in resp.content:
            xml_bytes = resp.content

    if not xml_bytes:
        return (
            None,
            None,
            "A chave foi lida, mas não conseguimos obter os dados do documento no momento "
            "(serviço indisponível ou documento inexistente).",
        )

    return xml_bytes, pdf_bytes, None


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

    # --- NF-e (55) e NFC-e (65) ---
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

    # --- CT-e (57) e CT-e OS (67) ---
    inf_cte = _buscar_primeiro(root, "infCte") or root
    rem_elem = _buscar_filho(inf_cte, "rem")
    dest_elem = _buscar_filho(inf_cte, "dest")
    exped_elem = _buscar_filho(inf_cte, "exped")
    receb_elem = _buscar_filho(inf_cte, "receb")

    remetente = _extrair_entidade(rem_elem)
    destinatario = _extrair_entidade(dest_elem)

    # Resolução do Tomador do Frete no CT-e
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
        # CT-e OS (modelo 67) possui a tag <toma> diretamente em <infCte>
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

    # Etapa 2: Consulta no endpoint público correspondente (ConsultaDanfe ou ConsultaDacte)
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
                "pdf_base64": None,
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