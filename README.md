# Analise-Multimodal-de-Sentimento
Sistema de Inteligência Artificial que analisa sentimento em vídeos combinando **três modalidades**: texto (transcrição), expressões faciais e tom de voz.
---
title: Análise Multimodal de Sentimento
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 🎬 Análise de Sentimento Multimodal

Sistema de Inteligência Artificial que analisa sentimento em vídeos combinando **três modalidades**: texto (transcrição), expressões faciais e tom de voz.

🔗 **Demo ao vivo:** https://huggingface.co/spaces/Danielfonseca1212/Multimodal

---

## 🎯 O que faz?

Analisa vídeos de reviews/depoimentos e determina o sentimento geral (positivo, negativo ou neutro) combinando:

| Modalidade | Tecnologia | O que analisa |
|------------|-----------|---------------|
| 📝 **Texto** | Whisper + BERT | Transcrição e sentimento das palavras |
| 😊 **Visual** | DeepFace | Expressões faciais e emoções |
| 🎵 **Áudio** | Librosa | Pitch, energia e prosódia da voz |

**Diferencial:** Detecta **inconsistências** entre o que a pessoa diz e sua expressão facial/tom de voz.

---

## 🚀 Como usar

1. **Acesse** o link do Space
2. **Faça upload** de um vídeo (máx 60 segundos, 100MB)
3. **Aguarde** o processamento (~30-60 segundos)
4. **Receba** a análise combinada das 3 modalidades

### 💡 Dicas para melhores resultados:
- Use vídeos com **rosto visível** e bem iluminado
- **Áudio claro** ajuda na transcrição
- Evite vídeos muito **escuros ou com muito ruído**

---## 🏗️ Arquitetura do Sistema

Vídeo de Entrada
│
├──→ 🔊 Áudio ──→ Whisper ──→ Texto ──→ BERT ──→ Sentimento Texto (40%)
│
├──→ 🎥 Frames ──→ DeepFace ──→ Expressões ──→ Sentimento Visual (35%)
│
└──→ 🎵 Áudio ──→ Librosa ──→ Features ──→ Sentimento Áudio (25%)
│
↓
┌─────────────────┐
│  Fusão Ponderada │ ← Detecta inconsistências
│  Weighted Voting │
└─────────────────┘
│
↓
Resultado Final + Explicação
plain
Copy

---

## 🛠️ Stack Tecnológico

| Categoria | Tecnologias |
|-----------|-------------|
| **Linguagem** | Python 3.13 |
| **Deep Learning** | PyTorch, Transformers (Hugging Face) |
| **Visão Computacional** | OpenCV, DeepFace |
| **Processamento de Áudio** | Librosa, SoundFile |
| **Vídeo** | MoviePy, ImageIO |
| **Deploy** | Gradio, Hugging Face Spaces |
| **ML Ops** | TensorFlow (backend DeepFace) |

### Modelos Utilizados:
- **Whisper Tiny** (OpenAI) - Transcrição de áudio
- **DistilBERT Multilíngue** - Análise de sentimento textual
- **RetinaFace + FER** (DeepFace) - Detecção facial e reconhecimento de emoções

---

## 📊 Exemplo de Resultado

### Input:
Vídeo de 15s de um cliente falando sobre um produto

### Output:
✅ POSITIVO | Score: 4.2/5
📊 Análise por Modalidade:
├─ 📝 Texto: POSITIVO (95% confiança)
│  "Adorei o produto, superou minhas expectativas..."
├─ 😊 Visual: POSITIVO (82% confiança, 12 frames)
│  Expressão: predominantemente 'happy'
└─ 🎵 Áudio: POSITIVO (75% confiança, 185Hz)
Pitch elevado indica entusiasmo
✅ Todas as modalidades concordam
plain
Copy

### Caso de Inconsistência:
⚠️ NEUTRO | Score: 3.1/5
📊 Análise por Modalidade:
├─ 📝 Texto: POSITIVO (88% confiança)
│  "O produto é ótimo, recomendo..."
├─ 😊 Visual: NEUTRO (45% confiança, 10 frames)
│  Expressão: predominantemente 'neutral'
└─ 🎵 Áudio: NEUTRO (60% confiança, 145Hz)
Tom de voz monótono
⚠️ ATENÇÃO: Inconsistência detectada!
O que foi dito (positivo) não corresponde à
expressão facial e tom de voz (neutros).
plain
Copy

---

## 🔬 Base Científica

Este sistema implementa técnicas de **Multimodal Fusion** baseadas em pesquisas recentes:

> "A fusão de múltiplas modalidades (texto, áudio, visual) supera a análise unimodal em 15-20% de acurácia na detecção de sentimentos."  
> — [MDPI Electronics, 2025](https://www.mdpi.com/2079-9292/14/20/4015)

**Técnica de fusão:** Weighted Voting Adaptativo  
- Peso inicial: Texto 40%, Visual 35%, Áudio 25%
- Ajuste dinâmico baseado na confiança de cada modalidade
- Detecção automática de inconsistências entre modalidades

---

## ⚙️ Instalação Local

```bash
# 1. Clone o repositório
git clone https://github.com/SEU_USUARIO/analise-sentimento-multimodal.git
cd analise-sentimento-multimodal

# 2. Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Instale dependências
pip install -r requirements.txt

# 4. Execute
python app.py
Acesse: http://localhost:7860
🚀 Deploy no Hugging Face Spaces
bash
Copy
# 1. Instale o CLI do Hugging Face
pip install huggingface-hub

# 2. Faça login
huggingface-cli login

# 3. Crie o Space
huggingface-cli repo create nome-do-space --type space --sdk gradio

# 4. Clone, adicione arquivos e push
git clone https://huggingface.co/spaces/SEU_USUARIO/nome-do-space
cd nome-do-space
# Copie app.py, requirements.txt, README.md
git add .
git commit -m "Deploy inicial"
git push
📈 Métricas e Performance
Table
Copy
Aspecto	Valor
Tempo de processamento	~30-60s para vídeo de 30s
Limite de tamanho	100MB (HF Spaces gratuito)
Limite de duração	60 segundos
Frames analisados	Até 20 frames por vídeo
Idiomas suportados	Português, Inglês (e outros via Whisper)
🎯 Casos de Uso
E-commerce: Análise automática de reviews em vídeo
Atendimento ao Cliente: Avaliação de chamadas de suporte
Pesquisa de Mercado: Análise de entrevistas de satisfação
Mídias Sociais: Monitoramento de sentimento em vídeos
Recursos Humanos: Análise de entrevistas de candidatos
🛡️ Limitações e Considerações
Table
Copy
Limitação	Explicação
Privacidade	Vídeos são processados em memória e não armazenados
Precisão	Dependente da qualidade do vídeo (iluminação, áudio)
Contexto cultural	Expressões faciais podem variar entre culturas
Vieses de modelo	Modelos pré-treinados podem conter vieses dos dados originais
🔧 Troubleshooting
Erro: "Vídeo muito longo"
Solução: Corte o vídeo para menos de 60 segundos usando:
bash
Copy
ffmpeg -i input.mp4 -t 30 -c copy output.mp4
Erro: "Nenhum rosto detectado"
Causa: Iluminação ruim ou rosto muito pequeno
Solução: Use vídeos com rosto próximo e bem iluminado
Erro: "Transcrição vazia"
Causa: Áudio muito baixo ou ruído excessivo
Solução: Verifique se o áudio está claro e sem música de fundo alta
🤝 Contribuição
Contribuições são bem-vindas! Para contribuir:
Fork o projeto
Crie uma branch (git checkout -b feature/nova-funcionalidade)
Commit suas mudanças (git commit -m 'Adiciona nova funcionalidade')
Push para a branch (git push origin feature/nova-funcionalidade)
Abra um Pull Request
Ideias para melhorias:
[ ] Suporte a mais idiomas
[ ] Análise de múltiplas pessoas no vídeo
[ ] Exportação de relatórios em PDF
[ ] Comparação lado a lado de dois vídeos
[ ] Dashboard com histórico de análises
📝 Licença
Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.
👨‍💻 Autor
[Seu Nome]
🔗 LinkedIn | 🐙 GitHub | 🌐 Portfólio
🙏 Agradecimentos
Hugging Face por disponibilizar modelos open-source
OpenAI pelo Whisper
Serengil pelo DeepFace
Comunidade open-source pelas ferramentas de ML
⭐ Se este projeto te ajudou, deixe uma star no repositório!
plain
Copy

---

**Substitua:**
- `SEU_USUARIO` pelo seu usuário do GitHub/HF
- `nome-do-space` pelo nome do seu Space
- `[Seu Nome]` pelo seu nome real
- Os links de LinkedIn/GitHub pelos seus

Quer que eu adicione mais alguma seção, como **"Roadmap"** ou **"Benchmarks comparativos"**?
https://huggingface.co/spaces/Danielfonseca1212/Multimodal  link do sistema atualize as informações
Edit
Copy
Share
Aqui está o README.md atualizado com o link do seu sistema:
Markdown
Copy
Code
Preview
---
title: Análise Multimodal de Sentimento
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 🎬 Análise de Sentimento Multimodal

Sistema de Inteligência Artificial que analisa sentimento em vídeos combinando **três modalidades**: texto (transcrição), expressões faciais e tom de voz.

🔗 **Demo ao vivo:** [https://huggingface.co/spaces/Danielfonseca1212/Multimodal](https://huggingface.co/spaces/Danielfonseca1212/Multimodal)

---

## 🎯 O que faz?

Analisa vídeos de reviews/depoimentos e determina o sentimento geral (positivo, negativo ou neutro) combinando:

| Modalidade | Tecnologia | O que analisa |
|------------|-----------|---------------|
| 📝 **Texto** | Whisper + BERT | Transcrição e sentimento das palavras |
| 😊 **Visual** | DeepFace | Expressões faciais e emoções |
| 🎵 **Áudio** | Librosa | Pitch, energia e prosódia da voz |

**Diferencial:** Detecta **inconsistências** entre o que a pessoa diz e sua expressão facial/tom de voz.

---

## 🚀 Como usar

1. **Acesse:** [https://huggingface.co/spaces/Danielfonseca1212/Multimodal](https://huggingface.co/spaces/Danielfonseca1212/Multimodal)
2. **Faça upload** de um vídeo (máx 60 segundos, 100MB)
3. **Aguarde** o processamento (~30-60 segundos)
4. **Receba** a análise combinada das 3 modalidades

### 💡 Dicas para melhores resultados:
- Use vídeos com **rosto visível** e bem iluminado
- **Áudio claro** ajuda na transcrição
- Evite vídeos muito **escuros ou com muito ruído**

---

## 🏗️ Arquitetura do Sistema
Vídeo de Entrada
│
├──→ 🔊 Áudio ──→ Whisper ──→ Texto ──→ BERT ──→ Sentimento Texto (40%)
│
├──→ 🎥 Frames ──→ DeepFace ──→ Expressões ──→ Sentimento Visual (35%)
│
└──→ 🎵 Áudio ──→ Librosa ──→ Features ──→ Sentimento Áudio (25%)
│
↓
┌─────────────────┐
│  Fusão Ponderada │ ← Detecta inconsistências
│  Weighted Voting │
└─────────────────┘
│
↓
Resultado Final + Explicação
plain
Copy

---

## 🛠️ Stack Tecnológico

| Categoria | Tecnologias |
|-----------|-------------|
| **Linguagem** | Python 3.13 |
| **Deep Learning** | PyTorch, Transformers (Hugging Face) |
| **Visão Computacional** | OpenCV, DeepFace |
| **Processamento de Áudio** | Librosa, SoundFile |
| **Vídeo** | MoviePy, ImageIO |
| **Deploy** | Gradio, Hugging Face Spaces |
| **ML Ops** | TensorFlow (backend DeepFace) |

### Modelos Utilizados:
- **Whisper Tiny** (OpenAI) - Transcrição de áudio
- **DistilBERT Multilíngue** - Análise de sentimento textual
- **RetinaFace + FER** (DeepFace) - Detecção facial e reconhecimento de emoções

---

## 📊 Exemplo de Resultado

### Input:
Vídeo de 15s de um cliente falando sobre um produto

### Output:
✅ POSITIVO | Score: 4.2/5
📊 Análise por Modalidade:
├─ 📝 Texto: POSITIVO (95% confiança)
│  "Adorei o produto, superou minhas expectativas..."
├─ 😊 Visual: POSITIVO (82% confiança, 12 frames)
│  Expressão: predominantemente 'happy'
└─ 🎵 Áudio: POSITIVO (75% confiança, 185Hz)
Pitch elevado indica entusiasmo
✅ Todas as modalidades concordam
plain
Copy

### Caso de Inconsistência:
⚠️ NEUTRO | Score: 3.1/5
📊 Análise por Modalidade:
├─ 📝 Texto: POSITIVO (88% confiança)
│  "O produto é ótimo, recomendo..."
├─ 😊 Visual: NEUTRO (45% confiança, 10 frames)
│  Expressão: predominantemente 'neutral'
└─ 🎵 Áudio: NEUTRO (60% confiança, 145Hz)
Tom de voz monótono
⚠️ ATENÇÃO: Inconsistência detectada!
O que foi dito (positivo) não corresponde à
expressão facial e tom de voz (neutros).
plain
Copy

---

## 🔬 Base Científica

Este sistema implementa técnicas de **Multimodal Fusion** baseadas em pesquisas recentes:

> "A fusão de múltiplas modalidades (texto, áudio, visual) supera a análise unimodal em 15-20% de acurácia na detecção de sentimentos."  
> — [MDPI Electronics, 2025](https://www.mdpi.com/2079-9292/14/20/4015)

**Técnica de fusão:** Weighted Voting Adaptativo  
- Peso inicial: Texto 40%, Visual 35%, Áudio 25%
- Ajuste dinâmico baseado na confiança de cada modalidade
- Detecção automática de inconsistências entre modalidades

---

## ⚙️ Instalação Local

```bash
# 1. Clone o repositório
git clone https://github.com/Danielfonseca1212/analise-sentimento-multimodal.git
cd analise-sentimento-multimodal

# 2. Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Instale dependências
pip install -r requirements.txt

# 4. Execute
python app.py
Acesse: http://localhost:7860
🚀 Deploy no Hugging Face Spaces
bash
Copy
# 1. Instale o CLI do Hugging Face
pip install huggingface-hub

# 2. Faça login
huggingface-cli login

# 3. Crie o Space
huggingface-cli repo create Multimodal --type space --sdk gradio

# 4. Clone, adicione arquivos e push
git clone https://huggingface.co/spaces/Danielfonseca1212/Multimodal
cd Multimodal
# Copie app.py, requirements.txt, README.md
git add .
git commit -m "Deploy inicial"
git push
📈 Métricas e Performance
Table
Copy
Aspecto	Valor
Tempo de processamento	~30-60s para vídeo de 30s
Limite de tamanho	100MB (HF Spaces gratuito)
Limite de duração	60 segundos
Frames analisados	Até 20 frames por vídeo
Idiomas suportados	Português, Inglês (e outros via Whisper)
🎯 Casos de Uso
E-commerce: Análise automática de reviews em vídeo
Atendimento ao Cliente: Avaliação de chamadas de suporte
Pesquisa de Mercado: Análise de entrevistas de satisfação
Mídias Sociais: Monitoramento de sentimento em vídeos
Recursos Humanos: Análise de entrevistas de candidatos
🛡️ Limitações e Considerações
Table
Copy
Limitação	Explicação
Privacidade	Vídeos são processados em memória e não armazenados
Precisão	Dependente da qualidade do vídeo (iluminação, áudio)
Contexto cultural	Expressões faciais podem variar entre culturas
Vieses de modelo	Modelos pré-treinados podem conter vieses dos dados originais
🔧 Troubleshooting
Erro: "Vídeo muito longo"
Solução: Corte o vídeo para menos de 60 segundos usando:
bash
Copy
ffmpeg -i input.mp4 -t 30 -c copy output.mp4
Erro: "Nenhum rosto detectado"
Causa: Iluminação ruim ou rosto muito pequeno
Solução: Use vídeos com rosto próximo e bem iluminado
Erro: "Transcrição vazia"
Causa: Áudio muito baixo ou ruído excessivo
Solução: Verifique se o áudio está claro e sem música de fundo alta
🤝 Contribuição
Contribuições são bem-vindas! Para contribuir:
Fork o projeto
Crie uma branch (git checkout -b feature/nova-funcionalidade)
Commit suas mudanças (git commit -m 'Adiciona nova funcionalidade')
Push para a branch (git push origin feature/nova-funcionalidade)
Abra um Pull Request
Ideias para melhorias:
[ ] Suporte a mais idiomas
[ ] Análise de múltiplas pessoas no vídeo
[ ] Exportação de relatórios em PDF
[ ] Comparação lado a lado de dois vídeos
[ ] Dashboard com histórico de análises
📝 Licença
Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.
👨‍💻 Autor
Daniel Fonseca
🔗 LinkedIn | 🐙 GitHub | 🚀 Hugging Face
🙏 Agradecimentos
Hugging Face por disponibilizar modelos open-source
OpenAI pelo Whisper
Serengil pelo DeepFace
Comunidade open-source pelas ferramentas de ML
⭐ Se este projeto te ajudou, deixe uma star no repositório!
plain
Copy

---

**Atualizações feitas:**
- ✅ Link do demo: `https://huggingface.co/spaces/Danielfonseca1212/Multimodal`
- ✅ Seu nome: **Daniel Fonseca**
- ✅ Seu usuário GitHub: `Danielfonseca1212`
- ✅ Seu perfil HF: `Danielfonseca1212`

**Para atualizar no HF Spaces:**
1. Edite o arquivo `README.md` no seu Space
2. Cole o conteúdo acima
3. Commit as mudanças

## 🏗️ Arquitetura do Sistema

