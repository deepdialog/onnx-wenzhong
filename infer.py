import os
import time

import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from onnxruntime import (
    GraphOptimizationLevel, InferenceSession,
    SessionOptions, get_all_providers
)
from transformers import GPT2Tokenizer


def create_model_for_provider(
    model_path: str,
    provider: str = 'CPUExecutionProvider'
) -> InferenceSession:
    assert provider in get_all_providers(), \
        f"provider {provider} not found, {get_all_providers()}"
    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = int(os.environ.get('NUM_THREADS', 16))
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session


def generate(
    text,
    max_len = 20,
    temperature = 0.95,
    top_p = 0.95,
    top_k = 50,
    eod=None,
    additional_eod=[],
    ban = []
):
    if eod is None:
        eod = [50256]
    input_ids = tokenizer(text)['input_ids']
    ids = []
    # kv_cache = np.zeros([30, 2, 1, 32, 1, 96]).astype(np.float32)
    kv_cache = kv_cache_start

    with tqdm() as pbar:
        for i in range(max_len):
            pbar.update(1)
            if i == 0:
                logits, kv_cache = model.run(['output', 'pkv_output'], {
                    "input": np.array([input_ids]).astype(np.int64),
                    'pkv': kv_cache
                })
            else:
                logits, kv_cache = model.run(['output', 'pkv_output'], {
                    "input": np.array([[next_token]], dtype=np.int64),
                    'pkv': kv_cache,
                })

            for x in ban:
                logits[:, -1, x] = -9999

            logits = logits / temperature
            scores = softmax(logits[:, -1, :])
            next_probs = np.sort(scores)[:, ::-1]
            if top_p > 0.0 and top_p < 1.0:
                next_probs = next_probs[:, :int(next_probs.shape[1] * (1 - top_p))]
            if top_k > 0 and top_k < next_probs.shape[1]:
                next_probs = next_probs[:, :top_k]
            next_probs_1 = next_probs / next_probs.sum(axis=1).reshape((-1, 1))

            next_tokens = np.argsort(scores)[:, ::-1]
            if top_p > 0.0 and top_p < 1.0:
                next_tokens = next_tokens[:, :int(next_tokens.shape[1] * (1 - top_p))]
            if top_k > 0 and top_k < next_tokens.shape[1]:
                next_tokens = next_tokens[:, :top_k]

            next_token = np.random.choice(next_tokens[0], p=next_probs_1[0])
            if eod is not None:
                if next_token in eod or next_token in additional_eod:
                    break
            ids.append(next_token)
    return text + tokenizer.decode(ids)


tokenizer = GPT2Tokenizer.from_pretrained('./tokenizer')
model = create_model_for_provider('./onnxq/model.onnx')
kv_cache_start = np.load('past_key_values.npy')
