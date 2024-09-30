import torch
from torchaudio.pipelines import SQUIM_OBJECTIVE
import torchaudio
#import evaluate
from transformers import (
    AutoModel,
    AutoProcessor,
    pipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperTokenizerFast,
)
from accelerate.utils.memory import release_memory
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
from scipy.spatial.distance import cosine

def clap_similarity(clap_model_name_or_path, texts, audios, device, input_sampling_rate=44100):
    clap = AutoModel.from_pretrained(clap_model_name_or_path)
    clap_processor = AutoProcessor.from_pretrained(clap_model_name_or_path)
    output_sampling_rate = clap_processor.feature_extractor.sampling_rate
    if input_sampling_rate != output_sampling_rate:
        audios = [
            torchaudio.functional.resample(torch.from_numpy(audio), input_sampling_rate, output_sampling_rate).numpy()
            for audio in audios
        ]
    clap_inputs = clap_processor(
        text=texts, audios=audios, padding=True, return_tensors="pt", sampling_rate=output_sampling_rate
    ).to(device)

    clap.to(device)
    with torch.no_grad():
        text_features = clap.get_text_features(
            clap_inputs["input_ids"], attention_mask=clap_inputs.get("attention_mask", None)
        )
        audio_features = clap.get_audio_features(clap_inputs["input_features"])

        cosine_sim = torch.nn.functional.cosine_similarity(audio_features, text_features, dim=1, eps=1e-8).mean()

    cosine_sim = cosine_sim.to("cpu")

    clap.to("cpu")
    clap, clap_inputs, audio_features, text_features = release_memory(clap, clap_inputs, audio_features, text_features)
    return cosine_sim


def si_sdr(audios, device, input_sampling_rate=44100):
    max_audio_length = 15 * SQUIM_OBJECTIVE.sample_rate
    model = SQUIM_OBJECTIVE.get_model().to((device))

    output_sampling_rate = SQUIM_OBJECTIVE.sample_rate
    if input_sampling_rate != output_sampling_rate:
        audios = [
            torchaudio.functional.resample(
                torch.tensor(audio)[None, :].to(device).float(), input_sampling_rate, output_sampling_rate
            )
            for audio in audios
        ]

    def apply_squim(waveform):
        with torch.no_grad():
            waveform = waveform[:, : min(max_audio_length, waveform.shape[1])]
            _, _, sdr_sample = model(waveform)
            sdr_sample = sdr_sample.cpu()[0]
        return sdr_sample

    si_sdrs = [apply_squim(audio) for audio in audios]
    audios, model = release_memory(audios, model)
    return si_sdrs


def wer(
    asr_model_name_or_path,
    prompts,
    audios,
    device,
    per_device_eval_batch_size,
    sampling_rate,
    noise_level_to_compute_clean_wer,
    si_sdr_measures,
):
    metric = evaluate.load("wer")
    asr_pipeline = pipeline(model=asr_model_name_or_path, device=device, chunk_length_s=25.0)

    return_language = None
    if isinstance(asr_pipeline.model, WhisperForConditionalGeneration):
        return_language = True

    transcriptions = asr_pipeline(
        [{"raw": audio, "sampling_rate": sampling_rate} for audio in audios],
        batch_size=int(per_device_eval_batch_size),
        return_language=return_language,
    )

    if isinstance(asr_pipeline.tokenizer, (WhisperTokenizer, WhisperTokenizerFast)):
        tokenizer = asr_pipeline.tokenizer
    else:
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

    english_normalizer = tokenizer.normalize
    basic_normalizer = tokenizer.basic_normalize

    normalized_predictions = []
    normalized_references = []

    for pred, ref in zip(transcriptions, prompts):
        normalizer = (
            english_normalizer
            if isinstance(pred.get("chunks", None), list) and pred["chunks"][0].get("language", None) == "english"
            else basic_normalizer
        )
        norm_ref = normalizer(ref)
        if len(norm_ref) > 0:
            norm_pred = normalizer(pred["text"])
            normalized_predictions.append(norm_pred)
            normalized_references.append(norm_ref)

    word_error = 100
    clean_word_error = None
    noisy_word_error = None
    percent_clean_samples = 0
    if len(normalized_references) > 0:
        word_error = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)
        

        if noise_level_to_compute_clean_wer and si_sdr_measures:
            si_sdr_measures = np.array(si_sdr_measures)
            mask = si_sdr_measures >= noise_level_to_compute_clean_wer
            if mask.any():
                clean_word_error = 100 * metric.compute(
                    predictions=np.array(normalized_predictions)[mask], references=np.array(normalized_references)[mask]
                )
                if not mask.all():
                    noisy_word_error = 100 * metric.compute(
                        predictions=np.array(normalized_predictions)[~mask], references=np.array(normalized_references)[~mask]
                    )
                else:
                    noisy_word_error = 0
                percent_clean_samples = mask.sum() / len(mask)

    asr_pipeline.model.to("cpu")
    asr_pipeline = release_memory(asr_pipeline)
    return word_error, [t["text"] for t in transcriptions], clean_word_error, noisy_word_error, percent_clean_samples

# Function to load and preprocess audio
def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio

# Function to extract speaker embedding
def extract_embedding(audio):
    # Step 1: Load Pre-trained Wav2Vec 2.0 Model and Processor
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
    model.eval()
    input_values = processor(audio, sampling_rate=16000, return_tensors='pt').input_values
    with torch.no_grad():
        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
    # Mean pooling
    embedding = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]
    embedding = embedding.squeeze().cpu().numpy()
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def speaker_similarity(generated_audio, 
                       reference_speaker_audio, 
                       threshold=0.8):
    if generated_audio is None or reference_speaker_audio is None:
        return None
    
    audio = load_audio(generated_audio)
    embedding = extract_embedding(audio)


    # Step 3: Verification Phase
    test_audio = load_audio(reference_speaker_audio)
    test_embedding = extract_embedding(test_audio)

    # Step 4: Compute Similarity Score
    similarity = np.dot(embedding, test_embedding)

    # Alternatively, use cosine distance (1 - cosine similarity)
    cosine_distance = cosine(embedding, test_embedding)
    similarity = 1 - cosine_distance  # Convert to similarity

    # Step 5: Decision Making
    #threshold = 0.8  # Set an appropriate threshold based on validation data
    if similarity > threshold:
        print("Speaker Verified: Match")
    else:
        print("Speaker Not Verified: No Match")

    print(f"Similarity Score: {similarity}")
    return similarity
