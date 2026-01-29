"""
TryOn Model.

Contains components adapted from FLUX.1 by Black Forest Labs (Apache-2.0):
https://github.com/black-forest-labs/flux
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from .utils import cast_tuple, compact, exists, unpack_images


# Use PyTorch's native scaled dot product attention (SDPA)
def _attn_processor(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Scaled dot product attention using PyTorch native implementation."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    x = _attn_processor(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


def prepare(img: Tensor, patch_size: int = 1) -> dict[str, Tensor]:
    bs, c, h, w = img.shape

    # Rearrange the image into patches based on the given patch size
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

    # Ensure all images in the batch are processed if the input batch size was 1
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    # Create image ids for positional encoding
    img_ids = torch.zeros(h // patch_size, w // patch_size, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // patch_size)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // patch_size)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    return img, img_ids.to(img.device)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        img_size = cast_tuple(img_size, 2)
        patch_size = cast_tuple(patch_size, 2)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if exists(norm_layer) else nn.Identity()

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                    These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = MLPEmbedder(frequency_embedding_size, hidden_size)
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size))

def apply_conditional_dropout(tensor, mask, null_tensor=None):
    device, dtype = tensor.device, tensor.dtype
    mask_shape = [mask.shape[0]] + [1] * (tensor.dim() - 1)
    keep_mask = mask.view(*mask_shape)

    if exists(null_tensor):
        null_tensor = null_tensor.to(device=device, dtype=dtype)
        null_tensor = null_tensor.expand_as(tensor)
    else:
        null_tensor = torch.zeros_like(tensor)

    return torch.where(keep_mask, tensor, null_tensor)


class TryOnModel(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int] = (864, 576),
        hidden_size: int = 1280,
        n_heads=10,
        double_blocks_depth: int = 8,
        single_blocks_depth: int = 16,
        mlp_ratio: int = 4,
        channels_in: int = 3,
        patch_size: int = 12,
        theta: int = 10000,
        axes_dim: Tuple[int] = (16, 56, 56),
        qkv_bias: bool = True,
        guidance_embed: bool = False,
        n_classes: int = 3,
        use_patch_mixer: bool = True,
        patch_mixer_depth: int = 4,
    ):
        super().__init__()

        # time
        self.t_embedder = TimestepEmbedder(hidden_size=hidden_size)

        # category labels (tops, bottoms, one-pieces)
        self.y_embedder = nn.Embedding(n_classes + 1, hidden_size) if n_classes > 0 else None  # +1 for null class

        # guidance embeddings for guidance distillation
        self.guidance_embedder = TimestepEmbedder(hidden_size=hidden_size) if guidance_embed else None

        # positional embeddings
        pe_dim = hidden_size // n_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)

        # images
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.channels_in = channels_in
        self.x_embedder = PatchEmbed(
            img_size=input_shape, patch_size=patch_size, in_chans=channels_in * 2 + 1, embed_dim=hidden_size, flatten=False
        )
        self.garment_embedder = PatchEmbed(
            img_size=input_shape, patch_size=patch_size, in_chans=channels_in + 1, embed_dim=hidden_size, flatten=False
        )

        # patch mixer
        self.use_patch_mixer = use_patch_mixer
        if use_patch_mixer:
            self.x_patch_mixer = nn.ModuleList(
                [SingleStreamBlock(hidden_size, n_heads, mlp_ratio=mlp_ratio) for _ in range(patch_mixer_depth)]
            )
            # Buffer kept for checkpoint compatibility (not used at inference)
            self.register_buffer("patch_mixer_token", torch.zeros(1, 1, channels_in * self.patch_size**2))

        # core MMDiT
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    hidden_size,
                    n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(double_blocks_depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(hidden_size, n_heads, mlp_ratio=mlp_ratio) for _ in range(single_blocks_depth)]
        )

        self.final_layer = LastLayer(hidden_size, out_channels=channels_in * self.patch_size**2)

    # forward with classifier free guidance

    def forward_for_cfg(self, *args, **kwargs):
        # cleanup kwargs
        kwargs = compact(kwargs)

        # infer batch size from the first tensor argument
        noisy_images = args[0]
        batch_size = noisy_images.shape[0]

        # duplicate all tensor arguments and keyword arguments
        duplicated_args = [torch.cat([arg, arg], dim=0) if isinstance(arg, torch.Tensor) else arg for arg in args]
        duplicated_kwargs = {
            k: torch.cat([v, v], dim=0) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
        }

        # prepare cond drop masks for the duplicated inputs
        mask = torch.cat(
            [
                torch.ones(batch_size, device=noisy_images.device, dtype=torch.bool),
                torch.zeros(batch_size, device=noisy_images.device, dtype=torch.bool),
            ],
            dim=0,
        )

        # add cond_drop_probs to duplicated_kwargs
        duplicated_kwargs["mask"] = mask

        # execute the forward pass with duplicated inputs
        all_logits = self.forward(*duplicated_args, **duplicated_kwargs)["x"]

        # split the logits into original and null versions
        logits, null_logits = all_logits.split(batch_size)

        return {"v_c": logits, "v_u": null_logits}

    def forward(
        self,
        x,
        times,
        ca_images,
        garment_images,
        person_poses,
        garment_poses,
        mask: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        garment_categories: Optional[torch.Tensor] = None,
    ):
        ###################### CLASSIFIER FREE GUIDANCE ######################

        batch_size, device = x.shape[0], x.device

        # if mask is not provided, create a boolean mask of all true
        if not exists(mask):
            mask = torch.ones(batch_size, device=device, dtype=torch.bool)

        ####################### 2D IMAGES TO SEQUENCE ########################

        ca_images = apply_conditional_dropout(ca_images, mask)
        person_poses = apply_conditional_dropout(person_poses, mask)
        x = torch.cat([x, ca_images, person_poses], dim=1)
        x = self.x_embedder(x)
        x, x_ids = prepare(x)

        garment_poses = apply_conditional_dropout(garment_poses, mask)
        garment_images = apply_conditional_dropout(garment_images, mask)
        garment_images = torch.cat([garment_images, garment_poses], dim=1)
        garment_images = self.garment_embedder(garment_images)
        garment_images, garment_ids = prepare(garment_images)

        ###################### TIME & MODULATION ######################

        t = self.t_embedder(times)

        if exists(self.guidance_embedder):
            assert exists(guidance), "Guidance scale required for guidance distilled model"
            t = t + self.guidance_embedder(guidance)

        if exists(self.y_embedder):
            assert exists(garment_categories), "Category labels required for y_embedder"
            y = apply_conditional_dropout(garment_categories, mask)
            t = t + self.y_embedder(y)

        ###################### POSITIONAL EMBEDDINGS ######################

        img, txt, vec = x, garment_images, t  # name change for consistency with the original code

        x_pe = self.pe_embedder(x_ids)
        g_pe = self.pe_embedder(garment_ids)

        ###################### PATCH MIXER ######################

        if self.use_patch_mixer:
            for block in self.x_patch_mixer:
                img = block(img, vec=vec, pe=x_pe)

        ###################### CORE MMDiT ########################

        pe = torch.cat([x_pe, g_pe], dim=2)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        x = self.final_layer(img, vec)

        ######################  SEQUENCE TO 2D IMAGES ########################

        x = rearrange(
            x,
            "b (h w) c -> b c h w",
            h=self.input_shape[0] // self.patch_size,
            w=self.input_shape[1] // self.patch_size,
        )
        if self.patch_size > 1:
            x = unpack_images(x, self.patch_size)

        return {"x": x}
