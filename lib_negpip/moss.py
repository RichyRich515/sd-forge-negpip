# MOSS-Image, as provided by https://github.com/eduardoabreu81/moss-image-forge-neo
#
# The extension registers its own diffusion engine (MossImageEngine), text engine
# (MossTextEngine: Qwen3.5-9B through Forge Neo's ComfyUI-compatible SDClipModel)
# and DiT (MossImageDiT: joint [image, text] single-stream blocks). None of them
# is part of Forge Neo, so everything here is resolved from the loaded model.

import json
import sys
from functools import wraps
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from scripts.negpip import NegPiP

    from modules.prompt_parser import SdConditioning

import torch
import torch.nn.functional as F

from backend import memory_management
from backend.args import dynamic_args
from backend.text_processing import emphasis, parsing
from lib_negpip.anima import _hook_compile_conditions
from lib_negpip.krea import _V_SCALING
from modules import shared

_PATCHED_MODEL = None
_PATCHED_DIT = None

# the pieces of the extension's transformer module the attention hook re-implements around
_TRANSFORMER_API = ("rotate", "head_norm", "situ", "kernels")


def is_moss_image(model) -> bool:
    return bool(getattr(model, "is_moss_image", False)) or type(model).__name__ == "MossImageEngine"


def patch_moss_negpip(cls: "NegPiP", *, unpatch=False) -> bool:
    global _PATCHED_MODEL, _PATCHED_DIT

    if unpatch != cls._patched[3]:
        return True

    if unpatch:
        if _PATCHED_MODEL is not None:
            _hook_get_learned_conditioning(_PATCHED_MODEL, True)
        if _PATCHED_DIT is not None:
            _hook_dit_forward(_PATCHED_DIT, True)
            _hook_block_forwards(_PATCHED_DIT, True)
        _hook_compile_conditions(True)

        _PATCHED_MODEL = None
        _PATCHED_DIT = None
        cls._patched[3] = False
        return True

    model = shared.sd_model
    dit = model.forge_objects.unet.model.diffusion_model

    problem = _unsupported(model, dit)
    if problem is not None:
        print(f"NegPiP Disabled (MOSS-Image: {problem})")
        return False

    _hook_get_learned_conditioning(model, False)
    _hook_dit_forward(dit, False)
    _hook_block_forwards(dit, False)
    _hook_compile_conditions(False)

    _PATCHED_MODEL = model
    _PATCHED_DIT = dit
    cls._patched[3] = True
    return True


def _unsupported(model, dit) -> Optional[str]:
    """a reason not to patch this MOSS-Image extension version, or None"""
    engine = getattr(model, "text_engine_moss", None)
    if engine is None:
        return "the engine has no text_engine_moss"
    engine_module = sys.modules.get(type(engine).__module__)
    for name in ("TEMPLATE", "MAX_LENGTH"):
        if not hasattr(engine_module, name):
            return f"the text engine module has no {name}"
    if not hasattr(engine, "token_ids") or not hasattr(engine, "encode"):
        return "the text engine has no token_ids / encode"
    if not hasattr(getattr(engine, "text_encoder", None), "encode_token_weights"):
        return "the text encoder has no encode_token_weights"
    if not hasattr(getattr(engine, "tokenizer", None), "tokenizer"):
        return "the tokenizer wrapper has no tokenizer"

    layers = getattr(dit, "layers", None)
    if not layers:
        return "the DiT has no layers"
    transformer_module = sys.modules.get(type(dit).__module__)
    for name in _TRANSFORMER_API:
        if not hasattr(transformer_module, name):
            return f"the transformer module has no {name}"
    block = layers[0]
    for name in ("norm1", "norm2", "q_norm", "k_norm", "proj", "heads", "fused_qkv"):
        if not hasattr(block, name):
            return f"the block has no {name}"
    return None


# ================================================================================ #


def _hook_get_learned_conditioning(model, remove: bool):
    if remove:
        if hasattr(model, "_negpip_orig_get_learned_conditioning"):
            if getattr(model.get_learned_conditioning, "_negpip", False):
                model.get_learned_conditioning = model._negpip_orig_get_learned_conditioning
            del model._negpip_orig_get_learned_conditioning
        return

    orig_get_learned_conditioning = model.get_learned_conditioning
    model._negpip_orig_get_learned_conditioning = orig_get_learned_conditioning

    engine = model.text_engine_moss

    @torch.inference_mode()
    @wraps(orig_get_learned_conditioning)
    def negpip_learned_conditioning(prompt: "SdConditioning"):
        memory_management.load_model_gpu(model.forge_objects.clip.patcher)
        v_scaling = _V_SCALING.get()

        # the extension hard-wires EmphasisNone; follow the user setting instead
        emphasis_name = emphasis.get_current_option(shared.opts.emphasis).name
        if any(emphasis.uses_emphasis(x) for x in prompt):
            dynamic_args.last_extra_generation_params["Emphasis"] = emphasis_name

        crossattn = []
        negpip_mask = []
        _count = 0
        cache = {}

        for line in prompt:
            if line not in cache:
                cache[line] = _encode_line(engine, line, emphasis_name, v_scaling)
            cond, mask = cache[line]

            _count += int((mask[..., -1] < 0).sum())

            crossattn.append(cond)
            negpip_mask.append(mask)

        if _count > 0:
            key = "Negative" if getattr(prompt, "is_negative_prompt", False) else "Positive"
            print(f"NegPiP Enable ({key}: {_count})")

        return {
            "crossattn": crossattn,
            "c_negpip_mask": negpip_mask,
        }

    negpip_learned_conditioning._negpip = True
    model.get_learned_conditioning = negpip_learned_conditioning


def _engine_module(engine):
    return sys.modules[type(engine).__module__]


def _is_layout(engine, text: str) -> bool:
    """
    a MOSS-Image layout prompt is one JSON object; its brackets are data, not
    emphasis, so such a line always goes through the extension's own tokenizer
    """
    restore = getattr(_engine_module(engine), "restore_layout_json", None)
    if restore is not None:
        text = restore(text)
    if not text.startswith("{"):
        return False
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


def _tokenize_line_negpip(engine, line: str, emphasis_name: str) -> tuple[list[int], list[float]]:
    """
    tokenize like MossTextEngine.token_ids (chat template once around the whole
    prompt), with the prompt weights attached to the tokens of their segments
    """
    text = line.strip()

    if _is_layout(engine, text):
        parsed = [(text, 1.0)]
    else:
        parsed = parsing.parse_prompt_attention(line, emphasis_name)
        if emphasis_name == "Ignore":
            parsed = [(t, 1.0) for t, _ in parsed]
        # MOSS-Image has no chunking; the parser marks BREAK with -1, keep it as
        # the plain word the extension would see
        parsed = [(t, 1.0 if (t == "BREAK" and w == -1.0) else w) for t, w in parsed]

    if all(w == 1.0 for _, w in parsed):
        tokens = engine.token_ids(line)
        return tokens, [1.0] * len(tokens)

    module = _engine_module(engine)
    prefix, suffix = module.TEMPLATE.split("{}")
    hf_tokenizer = engine.tokenizer.tokenizer

    texts = [t for t, _ in parsed]
    weights = [w for _, w in parsed]
    # token_ids() drops the whitespace around the whole prompt
    texts[0] = texts[0].lstrip()
    texts[-1] = texts[-1].rstrip()

    tokens = list(hf_tokenizer.encode(prefix, add_special_tokens=False))
    multipliers = [1.0] * len(tokens)

    for segment, weight in zip(texts, weights):
        if not segment:
            continue
        ids = hf_tokenizer.encode(segment, add_special_tokens=False)
        tokens.extend(ids)
        multipliers.extend([weight] * len(ids))

    ids = hf_tokenizer.encode(suffix, add_special_tokens=False)
    tokens.extend(ids)
    multipliers.extend([1.0] * len(ids))

    max_length = module.MAX_LENGTH
    return tokens[:max_length], multipliers[:max_length]


def _weight_masks(multipliers: list[float], v_scaling: float = 0.0) -> tuple[list[float], torch.Tensor]:
    """
    split the per-token prompt weights between the text encoder and the attention
    values, exactly as krea._encode_line does; returns (encoder_scales, mask) where
    mask is [tokens, 1] (sign only) or [tokens, 2] (image-query mask, sign mask)
    """
    encoder_fade = min(max(v_scaling, 0.0), 1.0)
    if encoder_fade >= 1.0:
        encoder_scales = [1.0] * len(multipliers)
    else:
        # Fade the ComfyUI-compatible encoder lerp out continuously as V-scaling
        # takes over, while retaining the exact endpoints at Strength 0 and 1.
        encoder_scales = [abs(m) + (1.0 - abs(m)) * encoder_fade for m in multipliers]

    weights = torch.tensor(multipliers, dtype=torch.float32)
    ones = torch.ones_like(weights)
    sign_mask = torch.where(weights < 0, -ones, ones)
    if v_scaling > 0.0:
        # Strength is an exponent: 0 gives the sign-only mask, 1 gives the raw
        # weight, and values above 1 strengthen magnitude without crossing zero.
        # Floor |w| so a zero weight still fades continuously from the sign-only
        # mask at Strength 0 instead of snapping to 0 for any Strength above it.
        image_mask = sign_mask * weights.abs().clamp_min(1e-4).pow(v_scaling)
        mask = torch.stack((image_mask, sign_mask), dim=-1)
    else:
        mask = sign_mask.unsqueeze(-1)

    return encoder_scales, mask


def _encode_line(engine, line: str, emphasis_name: str, v_scaling: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, multipliers = _tokenize_line_negpip(engine, line, emphasis_name)
    encoder_scales, mask = _weight_masks(multipliers, v_scaling)

    if all(w == 1.0 for w in multipliers):
        # identical to the extension's own encoding
        cond = engine.encode(line)
    else:
        # the ComfyUI-compatible encoder applies the magnitudes on its output, by
        # lerping each scaled token between the empty (pad) encoding and its own
        states = engine.text_encoder.encode_token_weights([list(zip(tokens, encoder_scales))])[0]
        cond = states[0]

    # (T, 4096) without a batch dimension, as the extension returns it
    if cond.ndim != 2:
        raise RuntimeError(f"Unexpected MOSS-Image conditioning shape: expected 2 dimensions, got {tuple(cond.shape)}")

    if mask.shape[0] < cond.shape[0]:
        mask = F.pad(mask, (0, 0, 0, cond.shape[0] - mask.shape[0]), value=1.0)
    elif mask.shape[0] > cond.shape[0]:
        mask = mask[: cond.shape[0]]

    return cond, mask.to(device=cond.device, dtype=cond.dtype)


# ================================================================================ #


def _hook_dit_forward(dit, remove: bool):
    if remove:
        if hasattr(dit, "_negpip_orig_forward"):
            if getattr(dit.forward, "_negpip", False):
                dit.forward = dit._negpip_orig_forward
            del dit._negpip_orig_forward
        return

    orig_forward = dit.forward
    dit._negpip_orig_forward = orig_forward

    @torch.inference_mode()
    @wraps(orig_forward)
    def negpip_forward(x, timestep, context, transformer_options=None, **kwargs):
        negpip_mask: Optional[torch.Tensor] = kwargs.pop("c_negpip_mask", None)

        if negpip_mask is not None:
            while negpip_mask.ndim > 3:
                negpip_mask = negpip_mask.squeeze(1)
            transformer_options = {**(transformer_options or {}), "negpip_mask": negpip_mask}

        return orig_forward(x, timestep, context, transformer_options=transformer_options, **kwargs)

    negpip_forward._negpip = True
    dit.forward = negpip_forward


def _hook_block_forwards(dit, remove: bool):
    for block in getattr(dit, "layers", ()):
        _hook_block_forward(block, remove)


def _hook_block_forward(block, remove: bool):
    if remove:
        if hasattr(block, "_negpip_orig_forward"):
            if getattr(block.forward, "_negpip", False):
                block.forward = block._negpip_orig_forward
            del block._negpip_orig_forward
        return

    orig_forward = block.forward
    block._negpip_orig_forward = orig_forward

    transformer = sys.modules[type(block).__module__]

    @torch.inference_mode()
    @wraps(orig_forward)
    def negpip_forward(x, rope, mod, dtype, transformer_options=None):
        options = transformer_options or {}
        negpip_mask: Optional[torch.Tensor] = options.get("negpip_mask", None)
        if negpip_mask is None:
            return orig_forward(x, rope, mod, dtype, transformer_options)

        # Block.forward of the extension, with the attention values of the text
        # tokens scaled by the NegPiP mask
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (m[:, None] for m in mod.unbind(1))
        y = (block.norm1(x) * (1 + scale_a) + shift_a).to(dtype)
        if block.fused_qkv:
            q, k, v = block.proj["qkv"](y).chunk(3, -1)
        else:
            q, k, v = block.proj["q"](y), block.proj["k"](y), block.proj["v"](y)
        q = transformer.head_norm(block.q_norm, q, block.heads)
        k = transformer.head_norm(block.k_norm, k, block.heads)
        out = _negpip_attention(transformer, q, k, v, rope, block.heads, negpip_mask, options)
        out = block.proj["o"](out.to(dtype))
        x = x + (out * gate_a).to(x.dtype)
        y = (block.norm2(x) * (1 + scale_f) + shift_f).to(dtype)
        hidden, gate = block.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + (block.proj["ffn_down"](transformer.situ(hidden, gate)) * gate_f).to(x.dtype)

    negpip_forward._negpip = True
    block.forward = negpip_forward


def _negpip_attention(transformer, q, k, v, rope, heads, negpip_mask, options) -> torch.Tensor:
    """transformer.attention() with the text values scaled; the text tokens follow the image tokens"""
    m = negpip_mask.to(v)
    if (batch := (v.size(0) // m.size(0))) > 1:
        m = m.repeat(batch, 1, 1)
    txtlen = min(m.size(1), v.size(1))
    imglen = v.size(1) - txtlen
    split_queries = m.size(-1) > 1
    image_mask = m[:, :txtlen, :1]
    sign_mask = m[:, :txtlen, -1:]

    # Text queries always read sign-masked values. When image queries need
    # different magnitudes, reuse this V tensor after text attention instead
    # of retaining a second head-expanded copy. (q and k are already projected
    # into fresh tensors, so writing into a chunk() view of qkv is safe.)
    v[:, imglen:] = v[:, imglen:] * sign_mask

    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = transformer.rotate(q.reshape(shape), rope).transpose(1, 2).to(v.dtype)
    k = transformer.rotate(k.reshape(shape), rope).transpose(1, 2).to(v.dtype)
    v = v.reshape(shape).transpose(1, 2)

    if not split_queries:
        return transformer.kernels.attention(q, k, v, transformer_options=options)

    out_txt = transformer.kernels.attention(q[:, :, imglen:], k, v, transformer_options=options)
    v[:, :, imglen:] = v[:, :, imglen:] * (image_mask * sign_mask).unsqueeze(1)
    out_img = transformer.kernels.attention(q[:, :, :imglen], k, v, transformer_options=options)
    return torch.cat((out_img, out_txt), dim=1)
