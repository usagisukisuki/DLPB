"""
ViT-Tiny with pluggable position encodings for LPB-V comparison experiments.

Position Encodings compared:
  no_pe       - no positional encoding
  ape         - learnable absolute position embedding (token-level, before blocks)
  alibi_2d    - 2D ALiBi: fixed linear distance bias in attention
  rpb         - Relative Position Bias table (Swin V1, per-layer learnable)
  cpb         - Continuous Position Bias MLP (Swin V2, per-layer)
  kerple_log_2d    - LPB-V isotropic: -r1*log(1+r2*dist), learnable r1/r2
  dlpb_aniso  - LPB-V anisotropic: + s*cos(phi - phi_star)
  dlpb_vm     - LPB-V von Mises: log-dist + 1st + 2nd order cos
  dlpb_vm3    - LPB-V vm + 3rd-order cos (120° periodic, corners/junctions)
  dlpb_ape_vm - Hybrid: APE token embedding + von Mises attention bias
  dlpb_movm   - LPB-V Mixture of von Mises (K=2): log-dist + beta*logsumexp(pi_k vM_k)
  dlpb_movm_sc - MoVM with per-head learnable scale alpha

DLPB variants (anisotropy folded into decay rate, not additive):
  dlpb_metric         - quadratic form M_h: B = -r1*log(1+r2*v^T M_h v), elliptic RF
  dlpb_vmf            - vMF single harmonic: g_h=exp(-κ*cos(2*(φ-ψ))), 2θ sharp
  dlpb_vmf_fourier    - vMF Fourier multi-harmonic: g_h=exp(-Σ κ_k cos(n_k(φ-μ_k)))
  *_sc                - scaled variants (per-head learnable α)
  *_rope_2d              - RoPE2D hybrid variants
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ============================================================
# Position Encoding Modules
# All implement:
#   apply_to_tokens(x) -> x   (token-level, only APE modifies)
#   get_attn_bias(N, device)  -> [1, H, N, N] or None
# ============================================================

class NoPE(nn.Module):
    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        return None

    def apply_to_qk(self, q, k):
        return q, k


class APE(nn.Module):
    """Learnable absolute position embedding added to patch tokens."""
    def __init__(self, num_tokens, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def apply_to_tokens(self, x):
        return x + self.pos_embed

    def get_attn_bias(self, N, device):
        return None

    def apply_to_qk(self, q, k):
        return q, k


class ALiBi2D(nn.Module):
    """2D ALiBi: fixed geometric slopes * euclidean distance."""
    def __init__(self, num_heads, grid_size):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor([2.0 ** (-8.0 * i / num_heads)
                                for i in range(1, num_heads + 1)])
        self.register_buffer('slopes', slopes)
        self.register_buffer('dist', _make_dist(grid_size))

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        d = self.dist[:N, :N].unsqueeze(0).unsqueeze(0)  # [1,1,N,N]
        return -self.slopes.view(1, -1, 1, 1) * d        # [1,H,N,N]

    def apply_to_qk(self, q, k):
        return q, k


class RPB(nn.Module):
    """Relative Position Bias table (Swin V1 style)."""
    def __init__(self, num_heads, grid_size):
        super().__init__()
        self.num_heads = num_heads
        table_len = (2 * grid_size - 1) ** 2
        self.table = nn.Parameter(torch.zeros(table_len, num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)
        self.register_buffer('idx', _make_rpb_idx(grid_size))

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        idx = self.idx[:N, :N]
        bias = self.table[idx.reshape(-1)].reshape(N, N, self.num_heads)
        return bias.permute(2, 0, 1).unsqueeze(0)  # [1,H,N,N]

    def apply_to_qk(self, q, k):
        return q, k


class CPB(nn.Module):
    """Continuous Position Bias MLP (Swin V2 style)."""
    def __init__(self, num_heads, grid_size):
        super().__init__()
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )
        self.register_buffer('rel_coords', _make_cpb_coords(grid_size))

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device, chunk_size=256):
        # Cache: bias depends only on N and model weights, not on input images
        cached = getattr(self, '_bias_cache', None)
        if cached is not None and cached.shape[2] == N:
            return cached
        coords = self.rel_coords[:N, :N]        # [N,N,2]
        # Chunked MLP to avoid [N,N,512] OOM at large resolutions
        chunks = []
        for i in range(0, N, chunk_size):
            c = 16 * torch.sigmoid(self.cpb_mlp(coords[i:i + chunk_size]))  # [c,N,H]
            chunks.append(c)
        bias = torch.cat(chunks, dim=0)          # [N,N,H]
        result = bias.permute(2, 0, 1).unsqueeze(0)  # [1,H,N,N]
        self._bias_cache = result
        return result

    def apply_to_qk(self, q, k):
        return q, k


class LPBVIsotropic(nn.Module):
    """LPB-V isotropic: -r1 * log(1 + r2 * dist)."""
    def __init__(self, num_heads, grid_size):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor([2.0 ** float(-i) for i in range(num_heads)])
        self.r1_raw = nn.Parameter(torch.log(torch.expm1(slopes.clamp(min=1e-6))))
        self.r2_raw = nn.Parameter(torch.zeros(num_heads))
        self.register_buffer('dist', _make_dist(grid_size))

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        d = self.dist[:N, :N].unsqueeze(0).unsqueeze(0)  # [1,1,N,N]
        r1 = F.softplus(self.r1_raw).view(1, -1, 1, 1)
        r2 = F.softplus(self.r2_raw).view(1, -1, 1, 1)
        return -r1 * torch.log1p(r2 * d)  # [1,H,N,N]

    def apply_to_qk(self, q, k):
        return q, k


class LPBVAnisotropic(nn.Module):
    """LPB-V anisotropic: log-distance + angular cos term."""
    def __init__(self, num_heads, grid_size):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor([2.0 ** float(-i) for i in range(num_heads)])
        self.r1_raw = nn.Parameter(torch.log(torch.expm1(slopes.clamp(min=1e-6))))
        self.r2_raw = nn.Parameter(torch.zeros(num_heads))
        phi_init = torch.linspace(0, 2 * math.pi, num_heads + 1)[:-1]
        self.phi_star = nn.Parameter(phi_init)
        self.s_raw = nn.Parameter(torch.full((num_heads,), -2.0))
        self.register_buffer('dist', _make_dist(grid_size))
        self.register_buffer('angle', _make_angle(grid_size))

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        d   = self.dist[:N, :N].unsqueeze(0).unsqueeze(0)   # [1,1,N,N]
        phi = self.angle[:N, :N].unsqueeze(0).unsqueeze(0)  # [1,1,N,N]

        r1 = F.softplus(self.r1_raw).view(1, -1, 1, 1)
        r2 = F.softplus(self.r2_raw).view(1, -1, 1, 1)
        dist_bias = -r1 * torch.log1p(r2 * d)

        s        = F.softplus(self.s_raw).view(1, -1, 1, 1)
        phi_star = self.phi_star.view(1, -1, 1, 1)
        ang_bias = s * torch.cos(phi - phi_star)

        return dist_bias + ang_bias  # [1,H,N,N]

    def apply_to_qk(self, q, k):
        return q, k


class LPBVVonMises(nn.Module):
    """LPB-V von Mises: log-distance + 1st-order cos + 2nd-order cos (180° periodic).

    The 2nd term models edge/line selectivity (orientation columns in V1):
      B = -r1*log(1+r2*r) + s1*cos(phi-phi1*) + s2*cos(2*(phi-phi2*))
    """
    def __init__(self, num_heads, grid_size):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor([2.0 ** float(-i) for i in range(num_heads)])
        self.r1_raw = nn.Parameter(torch.log(torch.expm1(slopes.clamp(min=1e-6))))
        self.r2_raw = nn.Parameter(torch.zeros(num_heads))
        # 1st order: single preferred direction (360° periodic)
        self.phi_star  = nn.Parameter(torch.linspace(0, 2 * math.pi, num_heads + 1)[:-1])
        self.s_raw     = nn.Parameter(torch.full((num_heads,), -2.0))
        # 2nd order: bi-directional edge selectivity (180° periodic)
        self.phi_star2 = nn.Parameter(torch.linspace(0, math.pi, num_heads + 1)[:-1])
        self.s_raw2    = nn.Parameter(torch.full((num_heads,), -2.0))
        self.register_buffer('dist',  _make_dist(grid_size))
        self.register_buffer('angle', _make_angle(grid_size))

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        d   = self.dist[:N, :N].unsqueeze(0).unsqueeze(0)   # [1,1,N,N]
        phi = self.angle[:N, :N].unsqueeze(0).unsqueeze(0)  # [1,1,N,N]

        r1 = F.softplus(self.r1_raw).view(1, -1, 1, 1)
        r2 = F.softplus(self.r2_raw).view(1, -1, 1, 1)
        dist_bias = -r1 * torch.log1p(r2 * d)

        s1       = F.softplus(self.s_raw).view(1, -1, 1, 1)
        phi_star = self.phi_star.view(1, -1, 1, 1)
        ang_bias1 = s1 * torch.cos(phi - phi_star)

        s2        = F.softplus(self.s_raw2).view(1, -1, 1, 1)
        phi_star2 = self.phi_star2.view(1, -1, 1, 1)
        ang_bias2 = s2 * torch.cos(2 * (phi - phi_star2))

        return dist_bias + ang_bias1 + ang_bias2  # [1,H,N,N]

    def apply_to_qk(self, q, k):
        return q, k


class LPBVVonMisesV3(nn.Module):
    """LPB-V vm3: log-distance + 1st + 2nd + 3rd-order cos.

    3rd term (120° periodic) captures corner/junction selectivity:
      B = -r1*log(1+r2*r)
        + s1*cos(phi-phi1*)
        + s2*cos(2*(phi-phi2*))
        + s3*cos(3*(phi-phi3*))
    """
    def __init__(self, num_heads, grid_size):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor([2.0 ** float(-i) for i in range(num_heads)])
        self.r1_raw = nn.Parameter(torch.log(torch.expm1(slopes.clamp(min=1e-6))))
        self.r2_raw = nn.Parameter(torch.zeros(num_heads))
        self.phi_star  = nn.Parameter(torch.linspace(0, 2 * math.pi, num_heads + 1)[:-1])
        self.s_raw     = nn.Parameter(torch.full((num_heads,), -2.0))
        self.phi_star2 = nn.Parameter(torch.linspace(0, math.pi, num_heads + 1)[:-1])
        self.s_raw2    = nn.Parameter(torch.full((num_heads,), -2.0))
        # 3rd order: 120° periodic (corners / tri-symmetric junctions)
        self.phi_star3 = nn.Parameter(torch.linspace(0, 2 * math.pi / 3, num_heads + 1)[:-1])
        self.s_raw3    = nn.Parameter(torch.full((num_heads,), -2.0))
        self.register_buffer('dist',  _make_dist(grid_size))
        self.register_buffer('angle', _make_angle(grid_size))

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        d   = self.dist[:N, :N].unsqueeze(0).unsqueeze(0)
        phi = self.angle[:N, :N].unsqueeze(0).unsqueeze(0)

        r1 = F.softplus(self.r1_raw).view(1, -1, 1, 1)
        r2 = F.softplus(self.r2_raw).view(1, -1, 1, 1)
        dist_bias = -r1 * torch.log1p(r2 * d)

        s1 = F.softplus(self.s_raw).view(1, -1, 1, 1)
        ang_bias1 = s1 * torch.cos(phi - self.phi_star.view(1, -1, 1, 1))

        s2 = F.softplus(self.s_raw2).view(1, -1, 1, 1)
        ang_bias2 = s2 * torch.cos(2 * (phi - self.phi_star2.view(1, -1, 1, 1)))

        s3 = F.softplus(self.s_raw3).view(1, -1, 1, 1)
        ang_bias3 = s3 * torch.cos(3 * (phi - self.phi_star3.view(1, -1, 1, 1)))

        return dist_bias + ang_bias1 + ang_bias2 + ang_bias3  # [1,H,N,N]

    def apply_to_qk(self, q, k):
        return q, k


# ============================================================
# Helper functions for precomputing coordinate tables
# ============================================================

def _make_coords(g):
    r = torch.arange(g).float()
    gy, gx = torch.meshgrid(r, r, indexing='ij')
    return torch.stack([gy.flatten(), gx.flatten()], dim=-1)  # [N,2]


def _make_dist(g):
    coords = _make_coords(g)
    diff = coords[:, None] - coords[None, :]  # [N,N,2]
    return diff.norm(dim=-1)                  # [N,N]


def _make_angle(g):
    coords = _make_coords(g)
    diff = coords[:, None] - coords[None, :]  # [N,N,2]
    return torch.atan2(diff[..., 1], diff[..., 0])  # [N,N]


def _make_rpb_idx(g):
    r = torch.arange(g)
    coords = torch.stack(torch.meshgrid(r, r, indexing='ij'), -1).reshape(-1, 2)
    rel = coords[:, None] - coords[None, :]   # [N,N,2]
    rel[..., 0] += g - 1
    rel[..., 1] += g - 1
    rel[..., 0] *= (2 * g - 1)
    return rel.sum(-1)  # [N,N]  flat index into (2g-1)^2 table


def _make_cpb_coords(g):
    r = torch.arange(g).float()
    coords = torch.stack(torch.meshgrid(r, r, indexing='ij'), -1).reshape(-1, 2)
    rel = coords[:, None] - coords[None, :]      # [N,N,2]
    rel = rel / max(g - 1, 1) * 8               # normalise to [-8, 8]
    return torch.sign(rel) * torch.log1p(rel.abs())  # [N,N,2] log-spaced


def _make_diff_components(g):
    """Precompute Δx², Δy², ΔxΔy for the quadratic-form DLPB.

    coords = (row, col) = (y, x); diff[i,j] = (yi-yj, xi-xj).
    Returns dxx=dx², dyy=dy², dxy=dx*dy, each [N, N].
    """
    coords = _make_coords(g)
    diff = coords[:, None] - coords[None, :]  # [N, N, 2]
    dy = diff[..., 0]
    dx = diff[..., 1]
    return dx * dx, dy * dy, dx * dy


# ============================================================
# LPB-V Mixture of von Mises (MoVM)
#
# B = -r1*log(1+r2*r) + beta * log sum_k pi_k exp(kappa_k cos(phi - mu_k))
#
# Simplified (Bessel normalization omitted); log-sum-exp stabilized via
# torch.logsumexp: log sum_k pi_k exp(x_k) = logsumexp(x_k + log pi_k, dim=K)
# ============================================================

class LPBVMoVM(nn.Module):
    """LPB-V Mixture of von Mises: log-dist + beta * logsumexp over K vM components.

    B = -r1*log(1+r2*r) + beta * log sum_k pi_k exp(kappa_k cos(phi - mu_k))

    No Bessel normalization (simple variant). Numerically stabilized via logsumexp.
    Params/head: 2 (r1,r2) + 1 (beta) + (K-1) (pi) + K (mu) + K (kappa) = 3K+2.
    """
    def __init__(self, num_heads, grid_size, K=2):
        super().__init__()
        self.num_heads = num_heads
        self.K = K
        slopes = torch.tensor([2.0 ** float(-i) for i in range(num_heads)])
        self.r1_raw    = nn.Parameter(torch.log(torch.expm1(slopes.clamp(min=1e-6))))
        self.r2_raw    = nn.Parameter(torch.zeros(num_heads))
        # beta: angular term global scale, init softplus(-2) ≈ 0.13
        self.beta_raw  = nn.Parameter(torch.full((num_heads,), -2.0))
        # pi logits: softmax over K gives mixing weights; init uniform (logits=0)
        self.pi_logits = nn.Parameter(torch.zeros(num_heads, K))
        # mu: center directions, spread uniformly across components and heads
        H = num_heads
        mu_init = torch.zeros(num_heads, K)
        for h in range(H):
            for k in range(K):
                mu_init[h, k] = 2 * math.pi * k / K + 2 * math.pi * h / (K * H)
        self.mu        = nn.Parameter(mu_init)
        # kappa: concentration >= 0, init softplus(0) = log2 ≈ 0.69
        self.kappa_raw = nn.Parameter(torch.zeros(num_heads, K))
        self.register_buffer('dist',  _make_dist(grid_size))
        self.register_buffer('angle', _make_angle(grid_size))

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        d   = self.dist[:N, :N].unsqueeze(0).unsqueeze(0)   # [1, 1, N, N]
        phi = self.angle[:N, :N].unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]

        r1 = F.softplus(self.r1_raw).view(1, -1, 1, 1)
        r2 = F.softplus(self.r2_raw).view(1, -1, 1, 1)
        dist_bias = -r1 * torch.log1p(r2 * d)

        beta  = F.softplus(self.beta_raw).view(1, -1, 1, 1)
        pi    = F.softmax(self.pi_logits, dim=-1)   # [H, K]
        kappa = F.softplus(self.kappa_raw)           # [H, K]
        mu    = self.mu                              # [H, K]

        # Expand [H,K] -> [K, H, 1, 1] for broadcasting with phi [1,1,N,N]
        pi_e    = pi.T.unsqueeze(-1).unsqueeze(-1)      # [K, H, 1, 1]
        mu_e    = mu.T.unsqueeze(-1).unsqueeze(-1)      # [K, H, 1, 1]
        kappa_e = kappa.T.unsqueeze(-1).unsqueeze(-1)   # [K, H, 1, 1]

        # x_k = kappa_k cos(phi - mu_k) + log(pi_k): [K, H, N, N]
        x = kappa_e * torch.cos(phi - mu_e) + torch.log(pi_e)
        # log-sum-exp over K (dim=0) -> [H, N, N]
        lse = torch.logsumexp(x, dim=0)

        return dist_bias + beta * lse.unsqueeze(0)   # [1, H, N, N]

    def apply_to_qk(self, q, k):
        return q, k


class LPBVMoVMScaled(LPBVMoVM):
    """LPBVMoVM with per-head learnable scale alpha."""
    def __init__(self, num_heads, grid_size, K=2):
        super().__init__(num_heads, grid_size, K)
        self.alpha = nn.Parameter(torch.ones(num_heads))

    def get_attn_bias(self, N, device):
        return self.alpha.view(1, -1, 1, 1) * super().get_attn_bias(N, device)


# ============================================================
# DLPB: Decomposed Log-Polar Bias
#
# Core idea (from design doc): fold anisotropy g_h(φ) into the
# decay rate rather than adding an angular term:
#
#   B(p_i, p_j) = -r1 * log(1 + r2 * g_h(φ_ij) * ||p_j - p_i||)
#
# Three implementations of g_h; all share the same properties:
#   · Elliptic oriented receptive field
#   · Δ(r,φ) = 0 at origin, saturates to -r1*log g_h at large r (auto)
#   · No artifacts at r=0 (log(1+0)=0)
#   · Resolution-extrapolation friendly (log-compressed)
# ============================================================

class DLPBMetric(nn.Module):
    """DLPB quadratic form (本命): B = -r1 * log(1 + r2 * v^T M_h v)

    M_h = L L^T, L = [[l1,0],[l3,l2]] — PSD guaranteed via Cholesky.
    Elliptical RF with 2θ symmetry; 5 params/head: r1, r2, l1, l2, l3.
    Init: l3=0, l1=l2=1 → isotropic (M_h = I).
    """
    def __init__(self, num_heads, grid_size):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor([2.0 ** float(-i) for i in range(num_heads)])
        self.r1_raw = nn.Parameter(torch.log(torch.expm1(slopes.clamp(min=1e-6))))
        self.r2_raw = nn.Parameter(torch.zeros(num_heads))
        init_l = math.log(math.e - 1)  # softplus(init_l) = 1.0
        self.l1_raw = nn.Parameter(torch.full((num_heads,), init_l))
        self.l2_raw = nn.Parameter(torch.full((num_heads,), init_l))
        self.l3     = nn.Parameter(torch.zeros(num_heads))  # off-diagonal, unconstrained
        dxx, dyy, dxy = _make_diff_components(grid_size)
        self.register_buffer('dxx', dxx)
        self.register_buffer('dyy', dyy)
        self.register_buffer('dxy', dxy)

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        l1 = F.softplus(self.l1_raw)           # [H]
        l2 = F.softplus(self.l2_raw)           # [H]
        l3 = self.l3                            # [H]
        M11 = (l1 * l1).view(-1, 1, 1)         # [H, 1, 1]
        M12 = (l1 * l3).view(-1, 1, 1)
        M22 = (l3 * l3 + l2 * l2).view(-1, 1, 1)
        # quadratic form q[h,i,j] = v^T M_h v, v = (dx, dy)
        q = M11 * self.dxx[:N, :N] + 2 * M12 * self.dxy[:N, :N] + M22 * self.dyy[:N, :N]
        r1 = F.softplus(self.r1_raw).view(1, -1, 1, 1)
        r2 = F.softplus(self.r2_raw).view(1, -1, 1, 1)
        return -r1 * torch.log1p(r2 * q.unsqueeze(0))   # [1, H, N, N]

    def apply_to_qk(self, q, k):
        return q, k


class DLPBvMFSingle(nn.Module):
    """DLPB vMF single harmonic: B = -r1 * log(1 + r2 * dist * g_h(phi))

    g_h(phi) = exp(-kappa * cos(2*(phi - psi))), 2θ periodicity.
    Sharper concentration than metric; 4 params/head: r1, r2, kappa, psi.
    Init: kappa≈0 → near-isotropic; psi spread over [0,π) across heads.
    """
    def __init__(self, num_heads, grid_size):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor([2.0 ** float(-i) for i in range(num_heads)])
        self.r1_raw    = nn.Parameter(torch.log(torch.expm1(slopes.clamp(min=1e-6))))
        self.r2_raw    = nn.Parameter(torch.zeros(num_heads))
        self.kappa_raw = nn.Parameter(torch.full((num_heads,), -3.0))  # softplus(-3)≈0.05
        self.psi       = nn.Parameter(torch.linspace(0, math.pi, num_heads + 1)[:-1])
        self.register_buffer('dist',  _make_dist(grid_size))
        self.register_buffer('angle', _make_angle(grid_size))

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        d   = self.dist[:N, :N].unsqueeze(0).unsqueeze(0)   # [1, 1, N, N]
        phi = self.angle[:N, :N].unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        r1    = F.softplus(self.r1_raw).view(1, -1, 1, 1)
        r2    = F.softplus(self.r2_raw).view(1, -1, 1, 1)
        kappa = F.softplus(self.kappa_raw).view(1, -1, 1, 1)
        psi   = self.psi.view(1, -1, 1, 1)
        g = torch.exp(-kappa * torch.cos(2 * (phi - psi)))
        return -r1 * torch.log1p(r2 * d * g)   # [1, H, N, N]

    def apply_to_qk(self, q, k):
        return q, k


class DLPBvMFFourier(nn.Module):
    """DLPB vMF Fourier: B = -r1 * log(1 + r2 * dist * g_h(phi))

    g_h(phi) = exp(-sum_k kappa_k * cos(n_k * (phi - mu_k)))
    Default harmonics (1, 2): 1θ directional + 2θ orientation selectivity.
    Params/head: 2 + 2*K (K = number of harmonics).
    Init: kappa_k≈0 → near-isotropic; mu_k spread uniformly across heads.
    """
    def __init__(self, num_heads, grid_size, harmonics=(1, 2)):
        super().__init__()
        self.num_heads = num_heads
        K = len(harmonics)
        slopes = torch.tensor([2.0 ** float(-i) for i in range(num_heads)])
        self.r1_raw    = nn.Parameter(torch.log(torch.expm1(slopes.clamp(min=1e-6))))
        self.r2_raw    = nn.Parameter(torch.zeros(num_heads))
        self.kappa_raw = nn.Parameter(torch.full((K, num_heads), -3.0))
        # Spread preferred directions uniformly across heads per harmonic
        mu_init = torch.zeros(K, num_heads)
        for k, n in enumerate(harmonics):
            period = 2 * math.pi / n
            mu_init[k] = torch.linspace(0, period, num_heads + 1)[:-1]
        self.mu = nn.Parameter(mu_init)
        self.register_buffer('ns',    torch.tensor(list(harmonics), dtype=torch.float32))
        self.register_buffer('dist',  _make_dist(grid_size))
        self.register_buffer('angle', _make_angle(grid_size))

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        d   = self.dist[:N, :N]    # [N, N]
        phi = self.angle[:N, :N]   # [N, N]
        r1    = F.softplus(self.r1_raw).view(1, -1, 1, 1)
        r2    = F.softplus(self.r2_raw).view(1, -1, 1, 1)
        kappa = F.softplus(self.kappa_raw)   # [K, H]
        ns    = self.ns                      # [K]
        mu    = self.mu                      # [K, H]
        # Broadcast: phi [N,N] → [1,1,N,N]; ns [K,1,1,1]; kappa,mu [K,H,1,1]
        phi_e = phi.unsqueeze(0).unsqueeze(0)         # [1, 1, N, N]
        ns_e  = ns.view(-1, 1, 1, 1)                  # [K, 1, 1, 1]
        mu_e  = mu.unsqueeze(-1).unsqueeze(-1)         # [K, H, 1, 1]
        k_e   = kappa.unsqueeze(-1).unsqueeze(-1)      # [K, H, 1, 1]
        log_g = -(k_e * torch.cos(ns_e * (phi_e - mu_e))).sum(0)  # [H, N, N]
        g     = torch.exp(log_g)
        d_e   = d.unsqueeze(0).unsqueeze(0)            # [1, 1, N, N]
        return -r1 * torch.log1p(r2 * d_e * g.unsqueeze(0))   # [1, H, N, N]

    def apply_to_qk(self, q, k):
        return q, k


# ============================================================
# DLPB Scaled variants (per-head α ablation)
# ============================================================

class DLPBMetricScaled(DLPBMetric):
    def __init__(self, num_heads, grid_size):
        super().__init__(num_heads, grid_size)
        self.alpha = nn.Parameter(torch.ones(num_heads))

    def get_attn_bias(self, N, device):
        return self.alpha.view(1, -1, 1, 1) * super().get_attn_bias(N, device)


class DLPBvMFSingleScaled(DLPBvMFSingle):
    def __init__(self, num_heads, grid_size):
        super().__init__(num_heads, grid_size)
        self.alpha = nn.Parameter(torch.ones(num_heads))

    def get_attn_bias(self, N, device):
        return self.alpha.view(1, -1, 1, 1) * super().get_attn_bias(N, device)


class DLPBvMFFourierScaled(DLPBvMFFourier):
    def __init__(self, num_heads, grid_size):
        super().__init__(num_heads, grid_size)
        self.alpha = nn.Parameter(torch.ones(num_heads))

    def get_attn_bias(self, N, device):
        return self.alpha.view(1, -1, 1, 1) * super().get_attn_bias(N, device)


# ============================================================
# ViT Building Blocks
# ============================================================

class Attention(nn.Module):
    def __init__(self, dim, num_heads, pe_module):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.pe = pe_module

    def forward(self, x):
        B, N, D = x.shape
        H, Hd = self.num_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, Hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, H, N, Hd]

        q, k = self.pe.apply_to_qk(q, k)

        scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        bias = self.pe.get_attn_bias(N, x.device)
        if bias is not None:
            scores = scores + bias

        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device).bernoulli_(keep) / keep
        return x * mask


class Block(nn.Module):
    def __init__(self, dim, num_heads, pe_module, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, pe_module)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ============================================================
# Full ViT-Tiny Model
# ============================================================

# ============================================================
# 2D Rotary Position Embedding
# ============================================================

class RoPE2D(nn.Module):
    """2D Rotary Position Embedding (Su et al. 2021, extended to 2D grids).

    head_dim is split in half: first half encodes row position, second half
    encodes column position.  Each half uses standard 1D RoPE frequencies.
    Requires head_dim % 4 == 0.
    """
    def __init__(self, num_heads, grid_size, head_dim, base=10000):
        super().__init__()
        assert head_dim % 4 == 0, f"head_dim ({head_dim}) must be divisible by 4 for RoPE2D"
        half = head_dim // 2      # dims devoted to each spatial direction
        d_pairs = half // 2       # number of frequency pairs per direction

        theta = 1.0 / (base ** (torch.arange(0, d_pairs).float() / d_pairs))  # [d_pairs]
        pos = torch.arange(grid_size).float()
        freqs = torch.outer(pos, theta)           # [G, d_pairs]
        cos_f = torch.cos(freqs)                  # [G, d_pairs]
        sin_f = torch.sin(freqs)

        N = grid_size * grid_size
        rows = torch.arange(N) // grid_size
        cols = torch.arange(N) % grid_size

        # Expand each pair so adjacent dims share the same frequency: [N, half]
        cos_row = cos_f[rows].repeat_interleave(2, dim=-1)
        sin_row = sin_f[rows].repeat_interleave(2, dim=-1)
        cos_col = cos_f[cols].repeat_interleave(2, dim=-1)
        sin_col = sin_f[cols].repeat_interleave(2, dim=-1)

        self.register_buffer('cos', torch.cat([cos_row, cos_col], dim=-1))  # [N, Hd]
        self.register_buffer('sin', torch.cat([sin_row, sin_col], dim=-1))

    @staticmethod
    def _rotate_half(x):
        """[..., d] → [..., d] mapping (x0,x1,x2,x3,...) to (-x1,x0,-x3,x2,...)."""
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        return torch.stack([-x_odd, x_even], dim=-1).flatten(-2)

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        return None

    def apply_to_qk(self, q, k):
        # q, k: [B, H, N, Hd]
        N = q.shape[2]
        cos = self.cos[:N].unsqueeze(0).unsqueeze(0)  # [1, 1, N, Hd]
        sin = self.sin[:N].unsqueeze(0).unsqueeze(0)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class RoPE2DHybrid(nn.Module):
    """Combines RoPE2D (Q/K rotation) with any additive attention-bias PE.

    apply_to_qk  → RoPE2D rotation
    get_attn_bias → delegated to the wrapped bias PE (e.g. LPBVVonMises)
    """
    def __init__(self, rope, bias_pe):
        super().__init__()
        self.rope = rope
        self.bias_pe = bias_pe

    def apply_to_tokens(self, x):
        return x

    def apply_to_qk(self, q, k):
        return self.rope.apply_to_qk(q, k)

    def get_attn_bias(self, N, device):
        return self.bias_pe.get_attn_bias(N, device)


# ============================================================
# Scaled LPB-V variants: output multiplied by learnable per-head α
#
# Purpose: ablation to disentangle "functional form" vs "magnitude"
#   α initialised to 1 so gradients flow from the start.
#   After training:
#     α → 0  ⟹  scale was the problem (CPB just learned a tiny bias)
#     α >> 0 but accuracy still low  ⟹  log-polar form itself is suboptimal
# ============================================================

class LPBVIsotropicScaled(LPBVIsotropic):
    def __init__(self, num_heads, grid_size):
        super().__init__(num_heads, grid_size)
        self.alpha = nn.Parameter(torch.ones(num_heads))

    def get_attn_bias(self, N, device):
        bias = super().get_attn_bias(N, device)          # [1, H, N, N]
        return self.alpha.view(1, -1, 1, 1) * bias


class LPBVAnisotropicScaled(LPBVAnisotropic):
    def __init__(self, num_heads, grid_size):
        super().__init__(num_heads, grid_size)
        self.alpha = nn.Parameter(torch.ones(num_heads))

    def get_attn_bias(self, N, device):
        bias = super().get_attn_bias(N, device)
        return self.alpha.view(1, -1, 1, 1) * bias


class LPBVVonMisesScaled(LPBVVonMises):
    def __init__(self, num_heads, grid_size):
        super().__init__(num_heads, grid_size)
        self.alpha = nn.Parameter(torch.ones(num_heads))

    def get_attn_bias(self, N, device):
        bias = super().get_attn_bias(N, device)
        return self.alpha.view(1, -1, 1, 1) * bias


class LPBVVonMisesV3Scaled(LPBVVonMisesV3):
    def __init__(self, num_heads, grid_size):
        super().__init__(num_heads, grid_size)
        self.alpha = nn.Parameter(torch.ones(num_heads))

    def get_attn_bias(self, N, device):
        bias = super().get_attn_bias(N, device)
        return self.alpha.view(1, -1, 1, 1) * bias


# ============================================================
# Fixed r1=r2=1 Scaled variants (ablation: remove distance-scale freedom)
# r1/r2 are not learned; bias = -log(1 + dist) * alpha
# ============================================================

class LPBVAnisotropicScaledFixed(nn.Module):
    """LPBVAnisotropicScaled with r1=r2=1 fixed (not learned)."""
    def __init__(self, num_heads, grid_size):
        super().__init__()
        self.num_heads = num_heads
        self.phi_star = nn.Parameter(torch.linspace(0, 2 * math.pi, num_heads + 1)[:-1])
        self.s_raw    = nn.Parameter(torch.full((num_heads,), -2.0))
        self.alpha    = nn.Parameter(torch.ones(num_heads))
        self.register_buffer('dist',  _make_dist(grid_size))
        self.register_buffer('angle', _make_angle(grid_size))

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        d   = self.dist[:N, :N].unsqueeze(0).unsqueeze(0)
        phi = self.angle[:N, :N].unsqueeze(0).unsqueeze(0)
        dist_bias = -torch.log1p(d)
        s        = F.softplus(self.s_raw).view(1, -1, 1, 1)
        phi_star = self.phi_star.view(1, -1, 1, 1)
        ang_bias = s * torch.cos(phi - phi_star)
        return self.alpha.view(1, -1, 1, 1) * (dist_bias + ang_bias)

    def apply_to_qk(self, q, k):
        return q, k


class LPBVVonMisesScaledFixed(nn.Module):
    """LPBVVonMisesScaled with r1=r2=1 fixed (not learned)."""
    def __init__(self, num_heads, grid_size):
        super().__init__()
        self.num_heads  = num_heads
        self.phi_star   = nn.Parameter(torch.linspace(0, 2 * math.pi, num_heads + 1)[:-1])
        self.s_raw      = nn.Parameter(torch.full((num_heads,), -2.0))
        self.phi_star2  = nn.Parameter(torch.linspace(0, math.pi, num_heads + 1)[:-1])
        self.s_raw2     = nn.Parameter(torch.full((num_heads,), -2.0))
        self.alpha      = nn.Parameter(torch.ones(num_heads))
        self.register_buffer('dist',  _make_dist(grid_size))
        self.register_buffer('angle', _make_angle(grid_size))

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        d   = self.dist[:N, :N].unsqueeze(0).unsqueeze(0)
        phi = self.angle[:N, :N].unsqueeze(0).unsqueeze(0)
        dist_bias = -torch.log1p(d)
        s1        = F.softplus(self.s_raw).view(1, -1, 1, 1)
        ang_bias1 = s1 * torch.cos(phi - self.phi_star.view(1, -1, 1, 1))
        s2        = F.softplus(self.s_raw2).view(1, -1, 1, 1)
        ang_bias2 = s2 * torch.cos(2 * (phi - self.phi_star2.view(1, -1, 1, 1)))
        return self.alpha.view(1, -1, 1, 1) * (dist_bias + ang_bias1 + ang_bias2)

    def apply_to_qk(self, q, k):
        return q, k


class LPBVVonMisesV3ScaledFixed(nn.Module):
    """LPBVVonMisesV3Scaled with r1=r2=1 fixed (not learned)."""
    def __init__(self, num_heads, grid_size):
        super().__init__()
        self.num_heads  = num_heads
        self.phi_star   = nn.Parameter(torch.linspace(0, 2 * math.pi, num_heads + 1)[:-1])
        self.s_raw      = nn.Parameter(torch.full((num_heads,), -2.0))
        self.phi_star2  = nn.Parameter(torch.linspace(0, math.pi, num_heads + 1)[:-1])
        self.s_raw2     = nn.Parameter(torch.full((num_heads,), -2.0))
        self.phi_star3  = nn.Parameter(torch.linspace(0, 2 * math.pi / 3, num_heads + 1)[:-1])
        self.s_raw3     = nn.Parameter(torch.full((num_heads,), -2.0))
        self.alpha      = nn.Parameter(torch.ones(num_heads))
        self.register_buffer('dist',  _make_dist(grid_size))
        self.register_buffer('angle', _make_angle(grid_size))

    def apply_to_tokens(self, x):
        return x

    def get_attn_bias(self, N, device):
        d   = self.dist[:N, :N].unsqueeze(0).unsqueeze(0)
        phi = self.angle[:N, :N].unsqueeze(0).unsqueeze(0)
        dist_bias = -torch.log1p(d)
        s1        = F.softplus(self.s_raw).view(1, -1, 1, 1)
        ang_bias1 = s1 * torch.cos(phi - self.phi_star.view(1, -1, 1, 1))
        s2        = F.softplus(self.s_raw2).view(1, -1, 1, 1)
        ang_bias2 = s2 * torch.cos(2 * (phi - self.phi_star2.view(1, -1, 1, 1)))
        s3        = F.softplus(self.s_raw3).view(1, -1, 1, 1)
        ang_bias3 = s3 * torch.cos(3 * (phi - self.phi_star3.view(1, -1, 1, 1)))
        return self.alpha.view(1, -1, 1, 1) * (dist_bias + ang_bias1 + ang_bias2 + ang_bias3)

    def apply_to_qk(self, q, k):
        return q, k


PE_CLASSES = {
    'no_pe':           (NoPE,                  {}),
    'ape':             (APE,                   {}),   # handled specially
    'alibi_2d':        (ALiBi2D,               {}),
    'rpb':             (RPB,                   {}),
    'cpb':             (CPB,                   {}),
    'kerple_log_2d':        (LPBVIsotropic,         {}),
    'dlpb_aniso':      (LPBVAnisotropic,       {}),
    'dlpb_vm':         (LPBVVonMises,          {}),
    'dlpb_vm3':        (LPBVVonMisesV3,        {}),
    'dlpb_ape_vm':     (LPBVVonMises,          {}),  # token PE handled in ViTTiny
    'kerple_log_2d_sc':     (LPBVIsotropicScaled,   {}),
    'dlpb':   (LPBVAnisotropicScaled, {}),
    'dlpb_O2':      (LPBVVonMisesScaled,    {}),
    'dlpb_O3':     (LPBVVonMisesV3Scaled,        {}),
    'dlpb_aniso_sc_fix': (LPBVAnisotropicScaledFixed, {}),
    'dlpb_vm_sc_fix':    (LPBVVonMisesScaledFixed,    {}),
    'dlpb_vm3_sc_fix':   (LPBVVonMisesV3ScaledFixed,  {}),
    # MoVM: Mixture of von Mises
    'dlpb_movm':         (LPBVMoVM,        {}),
    'dlpb_movm_sc':      (LPBVMoVMScaled,  {}),
    # RoPE2D (standalone and hybrid with LPBV bias)
    'rope_2d':          (RoPE2D,       {}),
    'kerple_log_2d_rope_2d':   (RoPE2DHybrid, {}),
    'dlpb_vm_rope_2d':    (RoPE2DHybrid, {}),
    'dlpb_vm3_rope_2d':   (RoPE2DHybrid, {}),
    'dlpb_rope_2d':(RoPE2DHybrid, {}),
    'dlpb_O2_rope_2d': (RoPE2DHybrid, {}),
    'dlpb_O3_rope_2d':(RoPE2DHybrid, {}),
    # RoPE2D + sc_fix variants
    'dlpb_aniso_sc_fix_rope_2d': (RoPE2DHybrid, {}),
    'dlpb_vm_sc_fix_rope_2d':    (RoPE2DHybrid, {}),
    'dlpb_vm3_sc_fix_rope_2d':   (RoPE2DHybrid, {}),
    # DLPB: Decomposed Log-Polar Bias (anisotropy inside the log)
    'dlpb_metric':             (DLPBMetric,          {}),
    'dlpb_vmf':                (DLPBvMFSingle,       {}),
    'dlpb_vmf_fourier':        (DLPBvMFFourier,      {}),
    'dlpb_metric_sc':          (DLPBMetricScaled,    {}),
    'dlpb_vmf_sc':             (DLPBvMFSingleScaled, {}),
    'dlpb_vmf_fourier_sc':     (DLPBvMFFourierScaled,{}),
    # DLPB + RoPE2D hybrids
    'dlpb_metric_rope_2d':        (RoPE2DHybrid, {}),
    'dlpb_vmf_rope_2d':           (RoPE2DHybrid, {}),
    'dlpb_vmf_fourier_rope_2d':   (RoPE2DHybrid, {}),
    'dlpb_metric_sc_rope_2d':     (RoPE2DHybrid, {}),
    'dlpb_vmf_sc_rope_2d':        (RoPE2DHybrid, {}),
    'dlpb_vmf_fourier_sc_rope_2d':(RoPE2DHybrid, {}),
}

PE_TYPES = list(PE_CLASSES.keys())

BASELINE_MODELS = ['resnet18', 'resnet50']
MODEL_TYPES = PE_TYPES + BASELINE_MODELS


class ViTTiny(nn.Module):
    """
    ViT-Tiny for CIFAR-100.
      image_size=32, patch_size=4 → grid 8×8 = 64 tokens
      embed_dim=192, depth=12, num_heads=3
    Uses global average pooling (no CLS token) for cleaner spatial PE.
    """
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=100,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        pe_type='ape',
    ):
        super().__init__()
        assert pe_type in PE_TYPES, f"Unknown pe_type '{pe_type}'. Choose from {PE_TYPES}"
        assert image_size % patch_size == 0
        self.pe_type = pe_type
        self.grid_size = image_size // patch_size
        num_tokens = self.grid_size ** 2

        # Patch embedding (conv shortcut, no bias to keep things clean)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim, patch_size, stride=patch_size, bias=False),
        )
        self.norm_pre = nn.LayerNorm(embed_dim)

        # Token-level PE: APE and the hybrid variant both add here
        if pe_type in ('ape', 'dlpb_ape_vm'):
            self.ape = APE(num_tokens, embed_dim)
        else:
            self.ape = None

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks, each with its own per-layer attention-bias PE
        head_dim = embed_dim // num_heads
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                pe_module=self._build_attn_pe(pe_type, num_heads, head_dim),
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _build_attn_pe(self, pe_type, num_heads, head_dim):
        """Build per-layer attention-bias PE module."""
        g = self.grid_size
        if pe_type in ('no_pe', 'ape'):
            return NoPE()
        elif pe_type == 'alibi_2d':
            return ALiBi2D(num_heads, g)
        elif pe_type == 'rpb':
            return RPB(num_heads, g)
        elif pe_type == 'cpb':
            return CPB(num_heads, g)
        elif pe_type == 'kerple_log_2d':
            return LPBVIsotropic(num_heads, g)
        elif pe_type == 'dlpb_aniso':
            return LPBVAnisotropic(num_heads, g)
        elif pe_type == 'dlpb_vm':
            return LPBVVonMises(num_heads, g)
        elif pe_type == 'dlpb_vm3':
            return LPBVVonMisesV3(num_heads, g)
        elif pe_type == 'dlpb_ape_vm':
            return LPBVVonMises(num_heads, g)
        elif pe_type == 'kerple_log_2d_sc':
            return LPBVIsotropicScaled(num_heads, g)
        elif pe_type == 'dlpb':
            return LPBVAnisotropicScaled(num_heads, g)
        elif pe_type == 'dlpb_O2':
            return LPBVVonMisesScaled(num_heads, g)
        elif pe_type == 'dlpb_O3':
            return LPBVVonMisesV3Scaled(num_heads, g)
        elif pe_type == 'dlpb_aniso_sc_fix':
            return LPBVAnisotropicScaledFixed(num_heads, g)
        elif pe_type == 'dlpb_vm_sc_fix':
            return LPBVVonMisesScaledFixed(num_heads, g)
        elif pe_type == 'dlpb_vm3_sc_fix':
            return LPBVVonMisesV3ScaledFixed(num_heads, g)
        elif pe_type == 'dlpb_movm':
            return LPBVMoVM(num_heads, g)
        elif pe_type == 'dlpb_movm_sc':
            return LPBVMoVMScaled(num_heads, g)
        elif pe_type == 'rope_2d':
            return RoPE2D(num_heads, g, head_dim)
        elif pe_type == 'kerple_log_2d_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), LPBVIsotropic(num_heads, g))
        elif pe_type == 'dlpb_vm_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), LPBVVonMises(num_heads, g))
        elif pe_type == 'dlpb_vm3_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), LPBVVonMisesV3(num_heads, g))
        elif pe_type == 'dlpb_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), LPBVAnisotropicScaled(num_heads, g))
        elif pe_type == 'dlpb_O2_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), LPBVVonMisesScaled(num_heads, g))
        elif pe_type == 'dlpb_O3_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), LPBVVonMisesV3Scaled(num_heads, g))
        elif pe_type == 'dlpb_aniso_sc_fix_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), LPBVAnisotropicScaledFixed(num_heads, g))
        elif pe_type == 'dlpb_vm_sc_fix_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), LPBVVonMisesScaledFixed(num_heads, g))
        elif pe_type == 'dlpb_vm3_sc_fix_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), LPBVVonMisesV3ScaledFixed(num_heads, g))
        # DLPB variants
        elif pe_type == 'dlpb_metric':
            return DLPBMetric(num_heads, g)
        elif pe_type == 'dlpb_vmf':
            return DLPBvMFSingle(num_heads, g)
        elif pe_type == 'dlpb_vmf_fourier':
            return DLPBvMFFourier(num_heads, g)
        elif pe_type == 'dlpb_metric_sc':
            return DLPBMetricScaled(num_heads, g)
        elif pe_type == 'dlpb_vmf_sc':
            return DLPBvMFSingleScaled(num_heads, g)
        elif pe_type == 'dlpb_vmf_fourier_sc':
            return DLPBvMFFourierScaled(num_heads, g)
        # DLPB + RoPE2D hybrids
        elif pe_type == 'dlpb_metric_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), DLPBMetric(num_heads, g))
        elif pe_type == 'dlpb_vmf_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), DLPBvMFSingle(num_heads, g))
        elif pe_type == 'dlpb_vmf_fourier_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), DLPBvMFFourier(num_heads, g))
        elif pe_type == 'dlpb_metric_sc_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), DLPBMetricScaled(num_heads, g))
        elif pe_type == 'dlpb_vmf_sc_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), DLPBvMFSingleScaled(num_heads, g))
        elif pe_type == 'dlpb_vmf_fourier_sc_rope_2d':
            return RoPE2DHybrid(RoPE2D(num_heads, g, head_dim), DLPBvMFFourierScaled(num_heads, g))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)              # [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)     # [B, N, C]
        x = self.norm_pre(x)

        if self.ape is not None:             # APE only
            x = self.ape.apply_to_tokens(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)                   # global average pooling
        return self.head(x)


def _build_resnet(arch, num_classes, image_size):
    model = tvm.resnet18(weights=None, num_classes=num_classes) if arch == 'resnet18' \
        else tvm.resnet50(weights=None, num_classes=num_classes)
    if image_size <= 64:
        # CIFAR adaptation: replace 7x7 stride-2 conv with 3x3 stride-1, remove maxpool
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model


def build_model(pe_type, image_size=32, patch_size=4, num_classes=100, drop_path_rate=0.1):
    if pe_type in BASELINE_MODELS:
        return _build_resnet(pe_type, num_classes, image_size)
    return ViTTiny(
        pe_type=pe_type,
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_pe_params(model):
    """Count only PE-related parameters (APE + per-layer attention bias PE)."""
    total = 0
    if model.ape is not None:
        total += sum(p.numel() for p in model.ape.parameters() if p.requires_grad)
    for block in model.blocks:
        total += sum(p.numel() for p in block.attn.pe.parameters() if p.requires_grad)
    return total


if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    for pe in PE_TYPES:
        m = build_model(pe)
        n = count_params(m)
        y = m(x)
        print(f"{pe:15s}  params={n/1e6:.3f}M  out={y.shape}")
