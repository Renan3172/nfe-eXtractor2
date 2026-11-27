# NF-eXtractor

> **Nova versão disponível:** Este projeto evoluiu para o [NF-eXtractor 2](https://github.com/Renan3172/nf-eXtractor2), com nova interface Neo-Brutalista, exibição de Remetente, Destinatário e Tomador do Frete diretamente na tela, extração de Número/Série/Modelo e suporte dedicado a CT-e.

Aplicação web para extração rápida da Chave de Acesso de Notas Fiscais Eletrônicas (NF-e) a partir de imagens (fotos ou capturas recebidas pelo WhatsApp) usando Google Gemini API, com consulta pública automática no ConsultaDanfe para download dos arquivos XML e PDF.

Identifica os 44 dígitos da chave de acesso a partir de imagens enviadas pelo navegador e permite colar imagens direto da área de transferência (como prints ou fotos copiadas do WhatsApp Web) sem precisar salvar o arquivo no computador.
Consulta e Download de NF-e: Busca automaticamente os dados da nota pela chave lida e disponibiliza o download dos arquivos XML e PDF (DANFE).
Arquitetura Leve: Sem banco de dados e sem armazenamento de histórico no navegador, mantendo a aplicação rápida e de baixa manutenção.

---

## Tecnologias:

* Backend: Python (api/index.py)
* Frontend: HTML5, CSS e JavaScript (public/index.html)
* Inteligência Artificial: Google Gemini API
* Consulta Fiscal: ConsultaDanfe
* Hospedagem: Vercel
