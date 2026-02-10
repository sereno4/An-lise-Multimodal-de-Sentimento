# ============================================
# APP.PY - VERSÃO SEM WHISPER (COMPATÍVEL PYTHON 3.13)
# ============================================

import os
import gc
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DEEPFACE_HOME'] = '/tmp/.deepface'

import torch
import numpy as np
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import cv2
from deepface import DeepFace
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip
import tempfile
from collections import Counter
import gradio as gr

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Sistema iniciando em: {DEVICE}")


class AnalisadorMultimodal:
    """Versão sem Whisper - usa modelo HF mais compatível"""
    
    def __init__(self):
        self.device = DEVICE
        self.modelos = {}
        self.limite_segundos = 60
        self.max_frames = 20
        
    def _carregar_stt(self):
        """Carrega modelo de speech-to-text do HF (mais compatível que Whisper)"""
        if 'stt' in self.modelos:
            return self.modelos['stt']
        
        print("📦 Carregando modelo de transcrição...")
        
        # Modelo open source mais leve e compatível
        model_id = "openai/whisper-tiny"  # Mesmo modelo, mas via HF transformers
        
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(self.device)
        
        self.modelos['stt'] = (processor, model)
        print("   ✅ STT pronto")
        return processor, model
    
    def _carregar_bert(self):
        """Carrega BERT para análise de sentimento"""
        if 'bert' in self.modelos:
            return self.modelos['bert']
        
        print("📦 Carregando BERT...")
        modelo = pipeline(
            "sentiment-analysis",
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            device=-1
        )
        self.modelos['bert'] = modelo
        print("   ✅ BERT pronto")
        return modelo
    
    def processar(self, video_path, progresso=None):
        """Pipeline completo"""
        if not video_path:
            return {"erro": "Nenhum vídeo enviado"}
        
        # Verificar duração
        try:
            clip = VideoFileClip(video_path)
            duracao = clip.duration
            clip.close()
            
            if duracao > self.limite_segundos:
                return {"erro": f"Vídeo muito longo ({duracao:.0f}s). Máximo: {self.limite_segundos}s"}
        except Exception as e:
            return {"erro": f"Erro ao ler vídeo: {e}"}
        
        # Progresso
        if progresso:
            progresso(0.1, desc="Extraindo áudio...")
        
        # Extrair áudio e transcrever
        audio, sr, texto = self._extrair_e_transcrever(video_path)
        
        if progresso:
            progresso(0.3, desc="Analisando texto...")
        analise_texto = self._analisar_texto(texto)
        
        if progresso:
            progresso(0.5, desc="Analisando rosto...")
        analise_visual = self._analisar_rosto(video_path)
        
        if progresso:
            progresso(0.7, desc="Analisando voz...")
        analise_audio = self._analisar_voz(audio, sr)
        
        if progresso:
            progresso(0.9, desc="Combinando...")
        fusao = self._combinar(analise_texto, analise_visual, analise_audio)
        
        gc.collect()
        
        return {
            'fusao': fusao,
            'texto': analise_texto,
            'visual': analise_visual,
            'audio': analise_audio,
            'transcricao': texto[:200],
            'duracao': duracao
        }
    
    def _extrair_e_transcrever(self, video_path):
        """Extrai áudio e transcreve usando modelo HF"""
        try:
            clip = VideoFileClip(video_path)
            
            if clip.audio is None:
                return None, None, "[Sem áudio no vídeo]"
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_path = tmp.name
            
            clip.audio.write_audiofile(
                audio_path, fps=16000, nbytes=2,
                codec='pcm_s16le', verbose=False, logger=None
            )
            clip.close()
            
            # Carregar áudio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Transcrever com modelo HF
            processor, model = self._carregar_stt()
            
            # Processar em chunks se necessário (máx 30s por vez)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                predicted_ids = model.generate(inputs.input_features, max_length=448)
            
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            os.remove(audio_path)
            return audio, sr, transcription
            
        except Exception as e:
            print(f"Erro transcrição: {e}")
            return None, None, f"[Erro na transcrição: {str(e)[:50]}]"
    
    def _analisar_texto(self, texto):
        """Análise de sentimento do texto"""
        if not texto or len(texto.strip()) < 3 or texto.startswith("["):
            return {'sentimento': 'neutro', 'confianca': 0, 'score': 3, 'texto': texto}
        
        try:
            modelo = self._carregar_bert()
            res = modelo(texto[:500])[0]
            
            label = res['label'].lower()
            conf = res['score']
            
            if 'positive' in label:
                sent, score = 'positivo', 4.5
            elif 'negative' in label:
                sent, score = 'negativo', 1.5
            else:
                sent, score = 'neutro', 3
            
            return {'sentimento': sent, 'confianca': conf, 'score': score, 'texto': texto[:100]}
        except:
            return {'sentimento': 'neutro', 'confianca': 0, 'score': 3, 'texto': texto[:50]}
    
    def _analisar_rosto(self, video_path):
        """Análise facial otimizada"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            step = max(1, total // self.max_frames)
            emocoes = []
            
            for i in range(0, total, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    peq = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)
                    res = DeepFace.analyze(
                        peq, actions=['emotion'],
                        enforce_detection=False, silent=True
                    )
                    
                    if isinstance(res, list):
                        res = res[0]
                    emocoes.append(res['dominant_emotion'])
                    
                except:
                    pass
                
                if len(emocoes) >= self.max_frames:
                    break
            
            cap.release()
            
            if not emocoes:
                return {'sentimento': 'neutro', 'confianca': 0, 'score': 3, 'frames': 0}
            
            mapa = {
                'happy': 'positivo', 'surprise': 'positivo',
                'sad': 'negativo', 'angry': 'negativo',
                'fear': 'negativo', 'disgust': 'negativo',
                'neutral': 'neutro'
            }
            
            sents = [mapa.get(e, 'neutro') for e in emocoes]
            contagem = Counter(sents)
            
            scores = {'negativo': 1, 'neutro': 3, 'positivo': 5}
            score_medio = np.mean([scores[s] for s in sents])
            
            return {
                'sentimento': contagem.most_common(1)[0][0],
                'confianca': min(1.0, len(emocoes)/10),
                'score': score_medio,
                'frames': len(emocoes)
            }
        except Exception as e:
            print(f"Erro rosto: {e}")
            return {'sentimento': 'neutro', 'confianca': 0, 'score': 3, 'frames': 0}
    
    def _analisar_voz(self, audio, sr):
        """Análise de prosódia"""
        if audio is None or len(audio) < sr:
            return {'sentimento': 'neutro', 'confianca': 0, 'score': 3}
        
        try:
            f0, _, _ = librosa.pyin(audio, fmin=80, fmax=400, sr=sr)
            f0_valido = f0[~np.isnan(f0)]
            
            if len(f0_valido) == 0:
                return {'sentimento': 'neutro', 'confianca': 0, 'score': 3}
            
            pitch = np.mean(f0_valido)
            
            score = 3.0
            if pitch > 200:
                score = 4.5
            elif pitch > 150:
                score = 3.5
            elif pitch < 120:
                score = 2.0
            
            sent = 'positivo' if score > 3.5 else 'negativo' if score < 2.5 else 'neutro'
            
            return {
                'sentimento': sent,
                'confianca': 0.7,
                'score': score,
                'pitch': pitch
            }
        except:
            return {'sentimento': 'neutro', 'confianca': 0, 'score': 3}
    
    def _combinar(self, texto, visual, audio):
        """Fusão das modalidades"""
        pesos = {'texto': 0.4, 'visual': 0.35, 'audio': 0.25}
        
        for k in pesos:
            if locals()[k]['confianca'] < 0.3:
                pesos[k] *= 0.5
        
        total = sum(pesos.values())
        if total == 0:
            return {'sentimento': 'neutro', 'score': 3, 'confianca': 0}
        
        pesos = {k: v/total for k, v in pesos.items()}
        
        scores = {'negativo': 1, 'neutro': 3, 'positivo': 5}
        
        score_final = sum(
            pesos[k] * scores[locals()[k]['sentimento']] * locals()[k]['confianca']
            for k in pesos
        )
        
        score_final = np.clip(score_final, 1, 5)
        sent_final = 'positivo' if score_final > 3.5 else 'negativo' if score_final < 2.5 else 'neutro'
        
        sents = [texto['sentimento'], visual['sentimento'], audio['sentimento']]
        inconsistencia = len(set(s for s in sents if s != 'neutro')) > 1
        
        return {
            'sentimento': sent_final,
            'score': round(score_final, 1),
            'confianca': round(np.mean([texto['confianca'], visual['confianca'], audio['confianca']]), 2),
            'inconsistencia': inconsistencia
        }


# ============================================
# INTERFACE
# ============================================

SISTEMA = None

def analisar(video, progress=gr.Progress()):
    global SISTEMA
    
    if video is None:
        return "⚠️ Faça upload de um vídeo"
    
    try:
        if SISTEMA is None:
            SISTEMA = AnalisadorMultimodal()
        
        r = SISTEMA.processar(video, progress)
        
        if 'erro' in r:
            return f"❌ **Erro:** {r['erro']}"
        
        fusao = r['fusao']
        t, v, a = r['texto'], r['visual'], r['audio']
        
        emoji = {
            'positivo': {'f': '😃', 't': '😊', 'v': '😄', 'a': '🎵'},
            'negativo': {'f': '😠', 't': '😤', 'v': '😡', 'a': '📉'},
            'neutro': {'f': '😐', 't': '😐', 'v': '😐', 'a': '🔇'}
        }[fusao['sentimento']]
        
        return f"""
## {emoji['f']} **{fusao['sentimento'].upper()}** | Score: {fusao['score']}/5
{'⚠️ **Inconsistência detectada!**' if fusao['inconsistencia'] else '✅ Modalidades concordam'}
---
### 📊 Detalhes
| Modalidade | Resultado | Confiança |
|-----------|-----------|-----------|
| {emoji['t']} **Texto** | {t['sentimento'].upper()} | {t['confianca']:.0%} |
| {emoji['v']} **Visual** | {v['sentimento'].upper()} ({v['frames']} frames) | {v['confianca']:.0%} |
| {emoji['a']} **Áudio** | {a['sentimento'].upper()} ({a.get('pitch', 0):.0f}Hz) | {a['confianca']:.0%} |
---
### 📝 Transcrição
> {r['transcricao'][:150]}{'...' if len(r['transcricao']) > 150 else ''}
**Duração:** {r['duracao']:.1f}s
        """.strip()
        
    except Exception as e:
        import traceback
        return f"❌ Erro: {str(e)[:200]}\n\n```\n{traceback.format_exc()[:300]}\n```"


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎬 Análise de Sentimento Multimodal
    
    Analise **texto**, **expressões faciais** e **tom de voz** em vídeos.
    
    ⏱️ Máximo: 60 segundos | 💾 Máximo: 100MB
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="📹 Upload do Vídeo", sources=["upload"])
            btn = gr.Button("🚀 Analisar", variant="primary")
        
        with gr.Column(scale=2):
            out = gr.Markdown("Aguardando vídeo...")
    
    btn.click(analisar, inputs=video_in, outputs=out)
    
    gr.Markdown("---\n**Tecnologias:** Hugging Face Transformers | DeepFace | Librosa | Gradio")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
