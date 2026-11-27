# NF-eXtractor 2

Aplicação web leve para leitura e extração rápida de dados de documentos fiscais(NF-e, NFC-e, CT-e e CT-e OS) a partir de imagens (fotos, capturas de tela ou arquivos recebidos pelo WhatsApp), combinando Reconhecimento Óptico de Caracteres (OCR) multimodal via Google Gemini API com consulta pública de NF-e e interface em estilo Neo-Brutalismo.

---

## Funcionalidades

* **Leitura de Chave de Acesso via IA:** Identifica os 44 dígitos da chave de acesso em fotos inclinadas, recortadas, com baixa iluminação ou impressas sem espaços.
* **Validação Matemática (Módulo 11) e Autocorreção:** Valida o Dígito Verificador da SEFAZ, o código de UF e o modelo do documento, corrigindo automaticamente erros comuns de contagem em sequências repetidas (como "000000" ou "1111").
* **Extração de Metadados Direto da Chave:** Exibe o Número (formatado em "N° 000.000.000"), Série e Modelo instantaneamente para todos os tipos de documentos suportados.
* **Consulta Automática de NF-e / NFC-e (Modelos 55 e 65):**
  * Busca o XML e o PDF em endpoints públicos do ConsultaDanfe.
  * Exibe Remetente, Destinatário e Tomador do Frete (tag <modFrete>) diretamente na tela.
  * Botões de cópia rápida que enviam apenas os 14 dígitos limpos do CNPJ (sem pontuação) para colagem direta em ERPs e emissores.
  * Download manual sob demanda dos arquivos XML e PDF.
* **Fluxo Dedicado para CT-e / CT-e OS (Modelos 57 e 67):**
  * Extrai e valida a chave de acesso e os metadados do documento (Número, Série e Modelo) com indicação visual de conferência de dados.
  * Exibe alerta informativo sobre a exigência de Certificado Digital A1 para download de XML/PDF de CT-e.
* **Produtividade e Privacidade:**
  * Suporte nativo ao "Ctrl + V" (colagem direta da área de transferência) e Drag & Drop.
  * Arquitetura stateless sem banco de dados: a chave da API do Gemini é armazenada exclusivamente no localStorage do navegador do usuário.
  * Layout otimizado para telas 1080p (sem rolagem) e totalmente responsivo para dispositivos móveis.

---

# Como utilizar

*1° Acesse: https://nf-extractor2.vercel.app/  
*2° Clique em Configurar API no canto superior direito, cole sua chave da Google Gemini API e clique em Salvar Chave (o status mudará para STATUS API GEMINI: OK).  
*3° Envie a imagem do documento fiscal clicando na área de upload, arrastando o arquivo ou pressionando Ctrl + V.  
*4° Clique em ENVIAR FOTO / EXTRAIR DOCUMENTO.  
*5° Utilize os botões COPIAR para levar a chave, o número do documento ou os CNPJs limpos direto para o seu sistema, ou clique em Baixar XML / Baixar PDF quando disponível.  

---

## Tecnologias Utilizadas

* Backend: Python, FastAPI, Requests, ElementTree (xml.etree.ElementTree)
* Frontend: HTML5, Tailwind CSS, JavaScript puro (Vanilla JS)
* Inteligência Artificial: Google Gemini API (com fallback automático entre modelos Flash e suporte a chaves AI Studio e Vertex AI `AQ....`)
* Hospedagem: Vercel (Serverless Functions)

---

nf-extractor-2/  
  
api/index.py - Backend FastAPI (OCR Gemini, Módulo 11, consulta NF-e e parser XML)  
public/index.html - Interface web  
requirements.txt - Dependências Python
