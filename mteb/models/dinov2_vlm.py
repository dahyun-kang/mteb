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


class DINOv2_VLM_Wrapper(Wrapper):
    def __init__(self, model_name: str, embed_dim: int | None = None, model_root='', **kwargs) -> None:
        from app.clip.models.lit.transformer import TextTransformer
        from app.clip.models.lit.tokenizer import tokenize as tokenizer

        self._model_name = model_name
        self._embed_dim = embed_dim
        if "768" in model_root:
            num_heads = 12
            num_layers = 12
            emb_dim = 768
            output_dim = 768
        else:
            num_heads = 20
            num_layers = 24
            emb_dim = 1280
            output_dim = 1280

        self.model = TextTransformer(
            context_length = 77,
            vocab_size = 49408,
            dim = emb_dim,
            num_heads = num_heads,
            num_layers = num_layers,
            mlp_ratio = 4.0,
            ls_init_value = None,  # layer scale initial value
            is_causal = True,
            pool_type = "argmax",
            output_tokens = False,
            output_dim=output_dim,
            dropout_prob = 0.0
        )
        self.tokenizer = tokenizer

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


text_embedding_clip_400k = ModelMeta(
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

text_embedding_clip_500K = ModelMeta(
    name="dinov2_vlm/ViT-L-14_768d12h12l_224_cls_50000_2000_32768_lr_0.0005_wd_0.2_768_dp_0.0_vb_0_vlp_False_beta1_0.9_beta2_0.98_eps_1e-06_fls_False_quickgelu_bf16",
    revision="",
    embed_dim=768+768, 
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

text_embedding_vb0_tdim768 = ModelMeta(
    name="dinov2_vlm/1101_metaclipv2mbraw_0vb_dim2048_tdim768_tlayer12_thead12_res224_50k_lr7e-4",
    revision="",
    embed_dim=2048, 
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



text_embedding_vb0_tdim768_32Kbatchsize_cls = ModelMeta(
    name="dinov2_vlm/vitlreg_768d12h12l_224_cls_50000_2000_32768_lr_0.0005_wd_0.0001_1024_dp_0.0_vb_0_vlp_False_beta1_0.9_beta2_0.99_eps_1e-08_fls_False",
    revision="",
    embed_dim=2048, 
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


text_embedding_vb0_tdim768_32Kbatchsize = ModelMeta(
    name="dinov2_vlm/vitlreg_768d12h12l_224_cls_patch0mean_50000_2000_32768_lr_0.0005_wd_0.0001_2048_dp_0.0_vb_0_vlp_False_beta1_0.9_beta2_0.99_eps_1e-08_fls_False",
    revision="",
    embed_dim=2048, 
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


text_embedding_vb2_tdim768 = ModelMeta(
    name="dinov2_vlm/1101_metaclipv2mbraw_2vb_dim2048_tdim768_tlayer12_thead12_res224_50k_lr7e-4",
    revision="",
    embed_dim=2048, 
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


text_embedding_vb0_tdim2048 = ModelMeta(
    name="dinov2_vlm/1101_4phases_3r_ver1_mul1_1500M_ssv2_0vb_dim2048_res224_50k_lr7e-4_sequential3",
    revision="",
    embed_dim=2048, 
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

text_embedding_vb2_tdim2048 = ModelMeta(
    name="dinov2_vlm/4phases_3r_ver1_mul1_1500M_ssv2_2vb_dim2048_res224_50k_lr7e-4_sequential3",
    revision="",
    embed_dim=2048, 
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


