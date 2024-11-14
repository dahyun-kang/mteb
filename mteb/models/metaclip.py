from __future__ import annotations

import os
import logging
import torch
from functools import partial
from typing import Any

import numpy as np

from mteb.model_meta import ModelMeta

from .wrapper import Wrapper
logger = logging.getLogger(__name__)


class VLM_Wrapper(Wrapper):
    def __init__(self, model_name: str, embed_dim: int | None = None, model_root='', **kwargs) -> None:
        from app.third_party.hf_meta_clip.setup import load_huggingface_meta_clip

        self._model_name = model_name
        self._embed_dim = embed_dim

        model, processor, tokenizer = load_huggingface_meta_clip('metaclip-l14-fullcc2.5b')
        self.model = model
        self.tokenizer = tokenizer

        self.model = self.model.cuda()

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:

        max_batch_size = 512
        sublists = [
            sentences[i : i + max_batch_size]
            for i in range(0, len(sentences), max_batch_size)
        ]

        all_embeddings = []

        for i, sublist in enumerate(sublists):
            tokens = self.tokenizer(sublist)
            print(f'tokens shape: {tokens.shape}')
            tokens = tokens.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                embeddings = self.model.encode_text(tokens)
                embeddings = embeddings.float().cpu().detach().numpy()
            print(f'{i}/{len(sublists)} ||| {sublist[0]} ||| {embeddings.shape}')

            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)


    def _to_numpy(self, embedding_response) -> np.ndarray:
        return np.array([e.embedding for e in embedding_response.data])

text_embedding_metaclip = ModelMeta(
    name="facebook/metaclip-l14-fullcc2.5b",
    revision="",
    embed_dim=1024, 
    # model_size='small',
    release_date="2024-11-03",
    languages=None,  # supported languages not specified
    loader=partial(VLM_Wrapper, model_name="large"),
    max_tokens=8191,
    open_weights=False,
    n_parameters=None,
    memory_usage=None,
    license=None,
    reference=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
)


