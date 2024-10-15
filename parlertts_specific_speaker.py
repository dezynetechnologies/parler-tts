import torch
import torch.nn as nn
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSAttention
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

#model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1").to(device)
#tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")
def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def _init_weights_cross_attention_2(module, module_name):
    if 'speaker_attn' in module_name and isinstance(module, ParlerTTSAttention):
        print(f"Initializing {module_name} with small random values")
        # Use Xavier uniform initialization with a small gain
        nn.init.xavier_uniform_(module.q_proj.weight, gain=1e-5)
        nn.init.xavier_uniform_(module.k_proj.weight, gain=1e-5)
        nn.init.xavier_uniform_(module.v_proj.weight, gain=1e-5)
        nn.init.xavier_uniform_(module.out_proj.weight, gain=1e-5)
        # Initialize biases to zero
        if module.q_proj.bias is not None:
            nn.init.zeros_(module.q_proj.bias)
        if module.k_proj.bias is not None:
            nn.init.zeros_(module.k_proj.bias)
        if module.v_proj.bias is not None:
            nn.init.zeros_(module.v_proj.bias)
        if module.out_proj.bias is not None:
            nn.init.zeros_(module.out_proj.bias)
    elif isinstance(module, nn.Linear) and 'speaker_embedding_projection_layer' in module_name:
        print(f"Initializing {module_name} with small random values")
        nn.init.xavier_uniform_(module.weight, gain=1e-5)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif 'speaker_attn_layer_norm' in module_name and isinstance(module, nn.LayerNorm):
        print(f"Initializing {module_name} with bias zeros and weight ones")
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _init_weights_cross_attention(module, module_name):
    if 'speaker_attn' in module_name and isinstance(module, ParlerTTSAttention):
        print(f"Initializing {module_name} with Xavier Uniform")
        nn.init.xavier_uniform_(module.q_proj.weight)
        nn.init.xavier_uniform_(module.k_proj.weight)
        nn.init.xavier_uniform_(module.v_proj.weight)
        nn.init.xavier_uniform_(module.out_proj.weight)      
    elif isinstance(module, nn.Linear) and 'speaker_embedding_projection_layer' in module_name:
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif 'speaker_attn_layer_norm' in module_name and isinstance(module, nn.LayerNorm):
        print(f"Initializing {module_name} with 0.0 bias and 1.0 weight")
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


for module_name, module in model.named_modules():
    _init_weights_cross_attention(module, module_name)

prompt = "Hey, how are you doing today? I hope you're having a great day!"
description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, reference_speaker = 'dheeraj_55s.mp3')
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out_partial_with_speech_token_24_layers_scaling_1x.wav", audio_arr, model.config.sampling_rate)