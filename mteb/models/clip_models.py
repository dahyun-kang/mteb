from __future__ import annotations

import logging
import torch
from functools import partial
from typing import Any

import numpy as np

from mteb.model_meta import ModelMeta

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


class CLIPWrapper(Wrapper):
    def __init__(self, model_name: str, embed_dim: int | None = None, **kwargs) -> None:

        self._model_name = model_name
        self._embed_dim = embed_dim

        import clip
        self.model, _ = clip.load("ViT-L/14", device='cpu')
        self.model = self.model.cuda()
        self.tokenizer = clip.tokenize

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
            embeddings = self.model.encode_text(tokens)
            embeddings = embeddings.float().cpu().detach().numpy()
            print(f'{i}/{len(sublists)} ||| {sublist[0]} ||| {embeddings.shape}')

            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)


    def _to_numpy(self, embedding_response) -> np.ndarray:
        return np.array([e.embedding for e in embedding_response.data])


class OpenCLIPWrapper(Wrapper):
    def __init__(self, model_name: str, embed_dim: int | None = None, **kwargs) -> None:

        self._model_name = model_name
        self._embed_dim = embed_dim

        import open_clip
        model, preprocess = open_clip.create_model_from_pretrained(f'hf-hub:{model_name}')
        tokenizer = open_clip.get_tokenizer(f'hf-hub:{model_name}')
        self.model = model.cuda()
        self.tokenizer = tokenizer

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
            embeddings = self.model.encode_text(tokens)
            embeddings = embeddings.float().cpu().detach().numpy()
            print(f'{i}/{len(sublists)} ||| {sublist[0]} ||| {embeddings.shape}')

            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)


    def _to_numpy(self, embedding_response) -> np.ndarray:
        return np.array([e.embedding for e in embedding_response.data])



text_embedding_3_clip = ModelMeta(
    name="clip",
    revision="",
    release_date="2024-11-03",
    languages=None,  # supported languages not specified
    loader=partial(CLIPWrapper, model_name="clip"),
    max_tokens=8191,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage=None,
    license=None,
    reference=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
)

text_embedding_open_clip = ModelMeta(
    name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    revision="",
    release_date="2024-11-03",
    languages=None,  # supported languages not specified
    loader=partial(OpenCLIPWrapper, model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K"),
    max_tokens=8191,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage=None,
    license=None,
    reference=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
)
