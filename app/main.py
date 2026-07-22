"""
Real-Time Voice & Emotion Processing - FastAPI backend.

Pipeline per audio chunk received over the /ws/audio WebSocket:
  1. Raw PCM16 mono audio bytes -> numpy float32 waveform.
  2. Real speech-to-text via faster-whisper (CTranslate2, CPU, int8).
  3. Real signal-derived emotion estimate via DSP features computed with
     librosa (RMS energy, pitch/F0 via YIN, zero-crossing rate, spectral
     centroid) mapped through a reasoned scoring rule-set (no randomness,
     no canned/fixed output - the label and confidence genuinely change
     with the acoustic characteristics of each chunk).
  4. Both results are pushed back to the browser immediately over the same
     WebSocket so the UI updates continuously while the user talks.
"""

import os
import json
import asyncio
import logging
import numpy as np
import librosa

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice-emotion")

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ---------------------------------------------------------------------------
# Whisper model (loaded once at startup, shared across connections)
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
logger.info("Loading faster-whisper model '%s' (CPU, int8)...", WHISPER_MODEL_SIZE)
# Network access to the HF CDN was previously blocked, so this was forced to
# local_files_only=True against a pre-cached 'small' model. That restriction
# has since been lifted; 'base' was downloaded and confirmed cached locally
# (see scratchpad test), so we now default to 'base' for better accuracy
# while still forcing local_files_only=True at startup so a future network
# hiccup can never hang server boot on a CDN connect-timeout.
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", local_files_only=True)
logger.info("Whisper model loaded.")

TARGET_SR = 16000  # sample rate whisper + our DSP features expect


def pcm16_bytes_to_float32(buf: bytes) -> np.ndarray:
    """Convert little-endian int16 PCM bytes to a float32 waveform in [-1, 1]."""
    if len(buf) < 2:
        return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(buf, dtype="<i2")
    return (arr.astype(np.float32) / 32768.0).copy()


def resample_if_needed(y: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR or len(y) == 0:
        return y
    return librosa.resample(y, orig_sr=src_sr, target_sr=TARGET_SR)


def transcribe_segments(y: np.ndarray):
    """Real speech-to-text using faster-whisper on a raw waveform buffer.

    Returns the list of decoded segments (each with .text/.start/.end) so
    the caller can reason about where speech actually ended within the
    buffer, instead of just a flat string.
    """
    if len(y) < TARGET_SR * 0.25:  # skip near-empty chunks
        return []
    segments, _info = whisper_model.transcribe(
        y,
        language="en",
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
        # Fixed temperature (no fallback re-decode ladder) keeps worst-case
        # latency bounded - noisy/non-speech chunks previously triggered
        # whisper's temperature-fallback retries and could take 60-90s+ on
        # CPU for a single 2.5s chunk instead of a couple seconds.
        temperature=0.0,
        no_speech_threshold=0.6,
    )
    return list(segments)


def _norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def analyze_emotion(y: np.ndarray, sr: int) -> dict:
    """
    Real, signal-derived emotion estimate. No randomness, no fixed output -
    every number below is computed from the actual audio chunk.

    Features:
      - RMS energy            -> loudness / arousal
      - F0 mean (YIN)         -> pitch height
      - F0 std (YIN)          -> pitch variability / expressiveness
      - Zero-crossing rate    -> noisiness / tension
      - Spectral centroid     -> brightness of the timbre

    These are combined with hand-reasoned weights into per-emotion scores,
    then turned into a probability distribution with a softmax so the UI
    always gets a full breakdown, not just a single label.
    """
    silence_result = {
        "emotion": "Neutral",
        "confidence": 0.35,
        "emotions": {"Neutral": 60, "Calm": 25, "Sad": 15},
        "features": {"rms": 0.0, "pitch_hz": 0.0, "pitch_std": 0.0, "zcr": 0.0, "centroid_hz": 0.0},
    }
    if len(y) < sr * 0.3:
        return silence_result

    rms = float(np.sqrt(np.mean(np.square(y))))
    if rms < 0.004:  # essentially silence / no speech
        return silence_result

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    try:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr)
        f0_voiced = f0[np.isfinite(f0) & (f0 > 0)]
        pitch_mean = float(np.mean(f0_voiced)) if f0_voiced.size else 0.0
        pitch_std = float(np.std(f0_voiced)) if f0_voiced.size else 0.0
    except Exception as exc:  # pragma: no cover
        logger.warning("pitch extraction failed: %s", exc)
        pitch_mean, pitch_std = 0.0, 0.0

    energy_n = _norm(rms, 0.005, 0.18)
    pitch_n = _norm(pitch_mean, 80, 300) if pitch_mean > 0 else 0.0
    pitch_var_n = _norm(pitch_std, 5, 55)
    zcr_n = _norm(zcr, 0.02, 0.18)
    bright_n = _norm(centroid, 700, 4000)

    scores = {
        "Excited": 0.35 * energy_n + 0.30 * pitch_n + 0.25 * pitch_var_n + 0.10 * bright_n,
        "Happy":   0.22 * energy_n + 0.23 * pitch_n + 0.20 * pitch_var_n + 0.35 * bright_n,
        "Stressed": 0.30 * energy_n + 0.15 * pitch_n + 0.35 * zcr_n + 0.20 * (1 - pitch_var_n),
        "Calm":    0.35 * (1 - energy_n) + 0.25 * (1 - zcr_n) + 0.25 * (1 - pitch_var_n) + 0.15 * (1 - bright_n),
        "Sad":     0.35 * (1 - energy_n) + 0.30 * (1 - pitch_n) + 0.20 * (1 - bright_n) + 0.15 * (1 - pitch_var_n),
        "Neutral": 0.45,
    }

    keys = list(scores.keys())
    vals = np.array([scores[k] for k in keys], dtype=np.float64)
    temp = 4.0
    exp = np.exp((vals - vals.max()) * temp)
    probs = exp / exp.sum()

    percentages = {k: round(float(p) * 100, 1) for k, p in zip(keys, probs)}
    percentages = dict(sorted(percentages.items(), key=lambda kv: -kv[1]))
    best = next(iter(percentages))
    confidence = round(percentages[best] / 100, 3)

    return {
        "emotion": best,
        "confidence": confidence,
        "emotions": percentages,
        "features": {
            "rms": round(rms, 4),
            "pitch_hz": round(pitch_mean, 1),
            "pitch_std": round(pitch_std, 1),
            "zcr": round(zcr, 4),
            "centroid_hz": round(centroid, 1),
        },
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


SILENCE_GAP_SECONDS = 0.6  # trailing gap with no recognized speech => commit + reset
MAX_BUFFER_SECONDS = 12.0  # hard cap so an uninterrupted speaker never balloons memory/latency
TRANSCRIBE_TIMEOUT = 20.0  # buffer can now hold up to MAX_BUFFER_SECONDS of audio, not just one chunk


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    client_sr = TARGET_SR
    chunk_index = 0
    # Per-connection rolling-buffer state: `buffer` accumulates raw 16kHz
    # audio since the last commit point, `committed_transcript` holds the
    # text already "locked in" at a detected pause/forced cut. Every chunk
    # re-transcribes the *whole* buffer (not just the new bytes) so words
    # never get split across chunk boundaries.
    buffer = np.zeros(0, dtype=np.float32)
    committed_transcript = ""
    logger.info("WebSocket client connected: %s", websocket.client)

    try:
        while True:
            message = await websocket.receive()

            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "start":
                    client_sr = int(payload.get("sampleRate", TARGET_SR))
                    logger.info("Client sample rate: %d Hz", client_sr)
                    await websocket.send_json({"type": "ready", "sampleRate": client_sr})
                elif payload.get("type") == "stop":
                    break
                continue

            if "bytes" in message and message["bytes"] is not None:
                raw = message["bytes"]
                if not raw:
                    continue
                chunk_index += 1

                y = pcm16_bytes_to_float32(raw)
                y16k = resample_if_needed(y, client_sr)
                if len(y16k) > 0:
                    buffer = np.concatenate([buffer, y16k])

                # Whisper + librosa are synchronous/CPU-bound; run them in a
                # worker thread so they never block the asyncio event loop
                # (blocking here would freeze every other WebSocket client
                # and even new incoming connection handshakes). A hard
                # timeout bounds worst-case latency per chunk so pathological
                # (e.g. pure noise) audio can never hang a connection.
                try:
                    segments = await asyncio.wait_for(
                        asyncio.to_thread(transcribe_segments, buffer), timeout=TRANSCRIBE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.warning("Transcription timed out for chunk %d; skipping text", chunk_index)
                    segments = []

                buffer_duration = len(buffer) / TARGET_SR
                full_text = " ".join(seg.text.strip() for seg in segments).strip()
                last_end = segments[-1].end if segments else 0.0
                trailing_silence = buffer_duration - last_end

                # A trailing gap with no recognized speech (or nothing
                # recognized at all) means the speaker paused: lock in
                # whatever was transcribed and start a fresh buffer.
                natural_pause = (not segments) or (trailing_silence >= SILENCE_GAP_SECONDS)
                forced_cut = (not natural_pause) and (buffer_duration > MAX_BUFFER_SECONDS) and segments
                committed_this_round = natural_pause or forced_cut

                if natural_pause:
                    if full_text:
                        committed_transcript = (committed_transcript + " " + full_text).strip()
                    buffer = np.zeros(0, dtype=np.float32)
                elif forced_cut:
                    # No natural pause yet, but the buffer is getting long -
                    # commit up through the last fully-decoded segment and
                    # keep only the still-undecoded tail so latency/memory
                    # stay bounded.
                    if full_text:
                        committed_transcript = (committed_transcript + " " + full_text).strip()
                    cut_sample = min(int(segments[-1].end * TARGET_SR), len(buffer))
                    buffer = buffer[cut_sample:].copy()
                # else: still mid-utterance, nothing committed yet.

                if committed_this_round:
                    # Buffer was just cleared/trimmed above; any leftover
                    # tail audio hasn't been transcribed yet and will show
                    # up on the next chunk, so don't double count `full_text`.
                    display_text = committed_transcript
                else:
                    display_text = (committed_transcript + " " + full_text).strip() if full_text else committed_transcript

                emotion = await asyncio.to_thread(analyze_emotion, y16k, TARGET_SR)

                await websocket.send_json(
                    {
                        "type": "result",
                        "chunk": chunk_index,
                        "text": display_text,
                        **emotion,
                    }
                )
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", websocket.client)
    except Exception as exc:  # pragma: no cover
        logger.exception("WebSocket error: %s", exc)
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5015)
