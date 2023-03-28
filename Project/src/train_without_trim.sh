python src/encoder/encoder_preprocess.py train --no_trim
python src/encoder/encoder_train.py no_trim train/SV2TTS/encoder
python src/synthesizer/synthesizer_preprocess_audio.py train
python src/synthesizer/synthesizer_preprocess_embeds.py train/SV2TTS/synthesizer
python src/synthesizer/synthesizer_train.py  no_trim train/SV2TTS/synthesizer
python src/vocoder/vocoder_preprocess.py train
python src/vocoder/vocoder_train.py no_trim train
