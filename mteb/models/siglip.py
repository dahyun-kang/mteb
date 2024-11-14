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


class SigLIP_Wrapper(Wrapper):
    def __init__(self, model_name: str, embed_dim: int | None = None, model_root='', **kwargs) -> None:
        from transformers import AutoProcessor, AutoModel
        if model_name == 'google/siglip-so400m-patch14-384':
            model_name = 0

        model = AutoModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)

        self._model_name = model_name
        self._embed_dim = embed_dim

        self.model = model
        self.tokenizer = processor

        # self.model = self.model.cuda()

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:

        max_batch_size = 512
        sublists = [
            sentences[i : i + max_batch_size]
            for i in range(0, len(sentences), max_batch_size)
        ]

        all_embeddings = []

        for i, sublist in enumerate(sublists):
            tokens = self.tokenizer(text=sublist, padding=True, truncation=True, return_tensors="pt")
            tokens = tokens['input_ids']
            # print(tokens)
            print(f'tokens shape: {tokens.shape}')
            # tokens = tokens.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                embeddings = self.model.text_model(tokens)
                embeddings = embeddings.last_hidden_state.mean(dim=1)
                embeddings = embeddings.float().cpu().detach().numpy()
            print(f'{i}/{len(sublists)} ||| {sublist[0]} ||| {embeddings.shape}')

            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)


    def _to_numpy(self, embedding_response) -> np.ndarray:
        return np.array([e.embedding for e in embedding_response.data])

text_embedding_siglip = ModelMeta(
    name="google/siglip-so400m-patch14-384",
    revision="",
    embed_dim=1024, 
    # model_size='small',
    release_date="2024-11-03",
    languages=None,  # supported languages not specified
    loader=partial(SigLIP_Wrapper, model_name="google/siglip-so400m-patch14-384"),
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


