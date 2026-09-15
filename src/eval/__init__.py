"""Evaluation package for RuLongTTS (role A4).

Modules:
    normalize    frozen text normalization (strict / lenient)
    alignment    global monotonic word alignment (jiwer backend)
    metrics      WER/CER/coverage/repetition/duration metrics
    asr_gigaam   GigaAM-v3 ASR wrapper (onnx-asr, VAD always on)
"""
