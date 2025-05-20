import os
import json
import subprocess
import tempfile
import shutil
from flask import Flask, request, send_file, render_template, jsonify
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Klasör ayarları
base_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(base_dir, "temp")
os.makedirs(temp_dir, exist_ok=True)

# Konfigürasyon yükleme
config_path = os.path.join(base_dir, "config.json")
config = {}
if os.path.exists(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            logger.info(f"Konfigürasyon yüklendi: {config}")
    except Exception as e:
        logger.error(f"Konfigürasyon dosyası okunamadı: {e}")
else:
    # Varsayılan konfigürasyon oluştur
    config = {
        "version": "2.1.0",
        "whisper_model": "medium",
        "cuda_enabled": True,
        "log_level": "INFO",
        "max_upload_size_mb": 10,
        "supported_languages": ["tr"],
        "temp_dir": "temp",
        "max_duration_sec": 60,
        "available_models": ["tiny", "base", "small", "medium", "large", "turbo"]
    }
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
            logger.info("Varsayılan konfigürasyon dosyası oluşturuldu")
    except Exception as e:
        logger.error(f"Varsayılan konfigürasyon oluşturulamadı: {e}")

# Tercih edilen model ve dil ayarları
preferred_model = config.get("whisper_model", "medium")
language = "tr"  # Varsayılan dil Türkçe
available_models = config.get("available_models", ["tiny", "base", "small", "medium", "large", "turbo"])

# Flask uygulamasını başlat
app = Flask(__name__, template_folder=os.path.dirname(os.path.abspath(__file__)),
            static_folder=os.path.dirname(os.path.abspath(__file__)))

# Whisper modeli için global değişken
whisper_model = None


def load_whisper_model(model_name=None):
    global whisper_model

    if model_name is None:
        model_name = preferred_model

    # Turbo modeli özel bir durum olarak ele alıyoruz
    if model_name == "turbo":
        try:
            # Faster-Whisper kullanarak turbo modeli yükleme
            from faster_whisper import WhisperModel

            logger.info(f"Turbo modeli yükleniyor (Faster-Whisper)")
            model_size = "medium"  # Turbo için "medium" modelini kullanıyoruz
            compute_type = "float16"  # Daha hızlı işlem için

            # Modeli yükle ve döndür
            model = WhisperModel(model_size, device="auto", compute_type=compute_type)
            logger.info(f"Turbo modeli başarıyla yüklendi: {model_size}")
            return model, "turbo"
        except ImportError as e:
            logger.warning(f"Faster-Whisper yüklenemedi, standart modele dönülüyor: {e}")
            model_name = "medium"  # Fallback to standard model
        except Exception as e:
            logger.error(f"Turbo model yüklenirken hata: {e}")
            model_name = "medium"  # Fallback to standard model

    # Standart OpenAI Whisper modeli
    if whisper_model is not None and isinstance(whisper_model, tuple) and whisper_model[1] == model_name:
        return whisper_model[0], whisper_model[1]

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available() and config.get("cuda_enabled", True):
            device = "cuda"
            logger.info(f"CUDA bulundu, GPU kullanılacak")
    except ImportError as e:
        logger.warning(f"PyTorch yüklenemedi, CPU kullanılacak: {e}")
    except Exception as e:
        logger.warning(f"CUDA kontrolü sırasında hata: {e}")

    try:
        import whisper
    except ImportError:
        logger.error("Whisper kütüphanesi yüklü değil")
        raise RuntimeError("Whisper kütüphanesi yüklü değil. Lütfen 'pip install openai-whisper' komutu ile kurun.")

    # Model yükleme sırası: istenen model, small, base, tiny
    model_options = [model_name, "small", "base", "tiny"]

    for model_option in model_options:
        try:
            logger.info(f"Whisper modeli yükleniyor: '{model_option}' (cihaz: {device})")
            model = whisper.load_model(model_option, device=device)
            logger.info(f"Whisper modeli başarıyla yüklendi: {model_option}")
            whisper_model = (model, model_option)
            return model, model_option
        except Exception as e:
            logger.error(f"Model {model_option} yüklenemedi: {e}")
            continue

    raise RuntimeError(
        "Hiçbir Whisper modeli yüklenemedi. Model dosyalarının mevcut olduğundan ve yeterli belleğiniz olduğundan emin olun.")


def cleanup_temp_files(file_path):
    """Geçici dosyaları temizle"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Geçici dosya silindi: {file_path}")
    except Exception as e:
        logger.error(f"Geçici dosya temizleme hatası: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """API durumunu kontrol etmek için endpoint"""
    return jsonify({
        "status": "running",
        "model": preferred_model,
        "language": language,
        "version": config.get("version", "2.1.0"),
        "available_models": available_models
    })


@app.route("/transcribe", methods=["POST"])
def transcribe():
    global whisper_model

    # İstemciden gelen model seçimi
    model_name = request.form.get("model", preferred_model)
    logger.info(f"İstek alındı: model={model_name}")

    if 'audio' not in request.files:
        logger.warning("Ses dosyası alınamadı")
        return jsonify({"error": "Ses dosyası alınamadı"}), 400

    file = request.files['audio']
    if file.filename == '':
        logger.warning("Dosya boş")
        return jsonify({"error": "Dosya boş"}), 400

    # Dosya adını ve uzantısını al
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext == '':
        ext = '.webm'

    # Desteklenen formatları kontrol et
    if ext not in ['.webm', '.wav', '.mp3', '.m4a', '.mp4', '.ogg']:
        logger.warning(f"Desteklenmeyen dosya formatı: {ext}")
        return jsonify({"error": f"Desteklenmeyen dosya formatı: {ext}"}), 400

    # Geçici dosya oluştur
    input_path = os.path.join(temp_dir, f"input_{os.urandom(8).hex()}{ext}")
    try:
        # Dosyayı geçici konuma kaydet
        file.save(input_path)
        logger.info(f"Ses dosyası kaydedildi: {input_path}")
    except Exception as e:
        logger.error(f"Dosya kaydedilemedi: {e}")
        cleanup_temp_files(input_path)
        return jsonify({"error": f"Dosya kaydedilemedi: {e}"}), 500

    # WAV formatına dönüştür
    output_wav = os.path.join(temp_dir, f"audio_{os.path.basename(input_path)}.wav")

    # FFmpeg yolunu kontrol et
    ffmpeg_path = os.path.join(base_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_path):
        ffmpeg_path = "ffmpeg"  # Sistem PATH'inden kullan

    # FFmpeg komutu
    cmd = [
        ffmpeg_path, "-y", "-i", input_path,
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        output_wav
    ]

    try:
        # FFmpeg'i çalıştır
        logger.info(f"FFmpeg komutu çalıştırılıyor: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            err_msg = result.stderr.strip()
            logger.error(f"FFmpeg hatası: {err_msg}")
            cleanup_temp_files(input_path)
            cleanup_temp_files(output_wav)
            return jsonify({"error": f"Ses formatı dönüştürülemedi: {err_msg}"}), 500

        logger.info("FFmpeg dönüşümü başarılı")
    except Exception as e:
        logger.error(f"FFmpeg çalıştırılamadı: {e}")
        cleanup_temp_files(input_path)
        cleanup_temp_files(output_wav)
        return jsonify({"error": f"FFmpeg çalıştırılamadı: {e}"}), 500

    try:
        # Seçilen modeli yükle
        model, model_used = load_whisper_model(model_name)

        # Turbo model özel işlemi
        if model_used == "turbo":
            # Faster-Whisper transcribe işlemi
            segments, info = model.transcribe(output_wav, language=language, beam_size=5)

            # Metni birleştir
            text = ""
            for segment in segments:
                text += segment.text + " "

            text = text.strip()
        else:
            # Standart Whisper transcribe işlemi
            logger.info(f"Ses dosyası metne çevriliyor: {output_wav}, model: {model_used}")
            result = model.transcribe(output_wav, language=language, verbose=False)
            text = result.get("text", "").strip()

        # Sonucu dosyaya kaydet
        transcription_path = os.path.join(temp_dir, "transcription.txt")
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(text)

        logger.info(f"Metne çevirme başarılı, model: {model_used}, karakter sayısı: {len(text)}")
    except Exception as e:
        logger.error(f"Metne çevirme başarısız: {e}")
        cleanup_temp_files(input_path)
        cleanup_temp_files(output_wav)
        return jsonify({"error": f"Metne çevirme başarısız: {e}"}), 500

    # Duygu analizi yap
    emotion_result, score = simple_emotion_analysis(text)

    # Geçici dosyaları temizle
    cleanup_temp_files(input_path)
    cleanup_temp_files(output_wav)

    return jsonify({
        "text": text,
        "emotion": emotion_result,
        "emotion_score": score,
        "model_used": model_used
    })


def simple_emotion_analysis(text):
    """Basit duygu analizi"""
    if not text:
        return "neutral", 0.5

    text_lower = text.lower()

    # Duygu kelimeleri - Genişletilmiş liste
    positive_words = [
        "iyi", "güzel", "harika", "mükemmel", "mutlu", "sevinçli", "seviyorum", "memnun", "pozitif",
        "teşekkür", "başarılı", "başardık", "sevindim", "keyifli", "neşeli", "muhteşem", "olağanüstü",
        "hoş", "sevgi", "mutluluk", "heyecan", "memnuniyet", "kutlama", "eğlence", "şahane",
        "muazzam", "kaliteli", "güvenli", "sağlıklı", "şanslı", "zengin", "rahat", "barış",
        "huzur", "dostluk", "gülümseme", "tebrik", "onur", "gurur", "kahkaha", "zevk", "sevimli",
        "tatlı", "şirin"
    ]

    negative_words = [
        "kötü", "berbat", "korkunç", "üzücü", "mutsuz", "sinirli", "kızgın", "nefret",
        "negatif", "üzgün", "başarısız", "kırgın", "endişeli", "kaygılı", "sıkıntılı",
        "dehşet", "panik", "acı", "ağrı", "keder", "mahzun", "bela", "sorun", "problem",
        "trajik", "öfke", "yalnızlık", "depresyon", "bunalım", "pişmanlık", "utanç", "suçluluk",
        "tedirgin", "hüzün", "çaresiz", "ızdırap", "yalancı", "aldatmak", "tehdit", "işkence",
        "bıktım", "isyan", "savaş", "hastalık", "endişe", "felaket", "faciaslı"
    ]

    # Kelime eşleşmelerini sayma
    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)

    # Detaylı log
    logger.info(f"Duygu analizi - Pozitif: {pos}, Negatif: {neg}, Metin uzunluğu: {len(text)}")

    # Sonucu hesapla
    if pos > neg:
        score = min(0.9, 0.5 + (pos * 0.1))
        logger.info(f"Duygu: Olumlu, Skor: {score:.2f}")
        return "happy", score
    elif neg > pos:
        score = max(0.1, 0.5 - (neg * 0.1))
        logger.info(f"Duygu: Olumsuz, Skor: {score:.2f}")
        return "sad", score
    else:
        logger.info("Duygu: Nötr, Skor: 0.50")
        return "neutral", 0.5


@app.route("/download", methods=["GET"])
def download_result():
    """Transkripsiyon metnini indirme"""
    result_file = os.path.join(temp_dir, "transcription.txt")

    if not os.path.exists(result_file):
        logger.warning("İndirilmek istenen transkripsiyon dosyası bulunamadı")
        return "Sonuç bulunamadı.", 404

    try:
        return send_file(result_file, as_attachment=True, download_name="transcription.txt")
    except Exception as e:
        logger.error(f"Dosya indirme hatası: {e}")
        return f"Dosya indirilemedi: {e}", 500


@app.errorhandler(500)
def internal_error(error):
    """500 hatalarını yakalama ve logla"""
    logger.error(f"500 sunucu hatası: {error}")
    return jsonify({"error": "Sunucu hatası. Lütfen daha sonra tekrar deneyin."}), 500


@app.errorhandler(404)
def not_found(error):
    """404 hatalarını yakalama"""
    logger.warning(f"404 hata: {request.path}")
    return jsonify({"error": "Sayfa bulunamadı."}), 404


if __name__ == "__main__":
    # Geçici klasörü temizle
    for file in os.listdir(temp_dir):
        try:
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Başlangıç temizliği hatası: {e}")

    # Sunucuyu başlat
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "True").lower() == "true"

    logger.info(f"Uygulama başlatılıyor: port={port}, debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)