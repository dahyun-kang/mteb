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
        from app.third_party.hf_meta_clip.setup import setup_and_build_model

        self._model_name = model_name
        self._embed_dim = embed_dim

        model, processor, tokenizer = load_huggingface_meta_clip('facebook/metaclip-l14-fullcc2.5b')
        self.model = model
        self.tokenizer = tokenizer

        '''
        print(kwargs)
        exp_root = root_dict[kwargs['revision']]  # '/checkpoint/dino/cijose/experiments/LiT/cvpr_main_experiments/MetaCLIP_v2_Balanced_090924_Mitigated_Airstore_access_modeITERABLE/'
        exp_name = reivision
        '''
        model_name = 'model_final.pth'
        model_path = os.path.join(model_root, model_name)
        model_ckpt = torch.load(model_path, map_location='cpu')

        ckpt_dict_text = {}

        for k in model_ckpt['model']:
            # '_orig_mod.text_model.text_projection.weight'[21:]
            if 'text_model' in k:
                ckpt_dict_text[k[21:]] = model_ckpt['model'][k]

        try:
            self.model.load_state_dict(ckpt_dict_text, strict=True)
        except RuntimeError:
            self.model.load_state_dict(ckpt_dict_text, strict=False)
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
                embeddings = self.model(tokens)
                embeddings = embeddings.float().cpu().detach().numpy()
            print(f'{i}/{len(sublists)} ||| {sublist[0]} ||| {embeddings.shape}')

            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)


    def _to_numpy(self, embedding_response) -> np.ndarray:
        return np.array([e.embedding for e in embedding_response.data])


text_embedding_clip = ModelMeta(
    name="dinov2_vlm",
    # exp_root="/checkpoint/dino/cijose/experiments/LiT/cvpr_main_experiments/MetaCLIP_v2_Balanced_090924_Mitigated_Airstore_access_modeITERABLE/",
    revision="ViT-L-14_768d12h12l_224_cls_400000_2000_32768_lr_0.0005_wd_0.2_768_dp_0.0_vb_0_vlp_False_beta1_0.9_beta2_0.98_eps_1e-06_fls_False_quickgelu_bf16",
    embed_dim=768, 
    # model_size='small',
    release_date="2024-11-03",
    languages=None,  # supported languages not specified
    loader=partial(DINOv2_VLM_Wrapper, model_name="large"),
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

text_embedding_metaclip = ModelMeta(
    name="facebook/metaclip-l14-fullcc2.5b",
    revision="",
    embed_dim=1024, 
    # model_size='small',
    release_date="2024-11-03",
    languages=None,  # supported languages not specified
    loader=partial(DINOv2_VLM_Wrapper, model_name="large"),
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


