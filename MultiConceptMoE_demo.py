# Based on https://github.com/ZihaoHuang-notabot/ConceptMoE
import torch
import torch.nn.functional as F
from torch import nn
from einops import repeat, rearrange

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except:
    print("mamba_chunk_scan_combined not found")
    mamba_chunk_scan_combined = None


def sharpen_prob_power(p, sharp_rate):
    """
    使用幂次变换
    sharp_rate > 1: 更尖锐
    sharp_rate = 1: 不变
    sharp_rate < 1: 更平滑
    """
    # 对于p > 0.5，使用 p^(1/sharp_rate)
    # 对于p < 0.5，使用 1 - (1-p)^(1/sharp_rate)
    sharpened_p = torch.where(
        p > 0.5,
        torch.pow(p, 1.0 / sharp_rate),
        1.0 - torch.pow(1.0 - p, 1.0 / sharp_rate)
    )
    return sharpened_p

class ChunkModule(nn.Module):
    def __init__(self, hidden_size=2048, compress_ratio=2):
        super().__init__()
        self.q_proj_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R = compress_ratio
        self.bound_prob_sharp_rate = 6

    def forward(self, hidden_states, aux_token_mask=None):
        cos_sim = torch.einsum(
                    "b l d, b l d -> b l",
                    F.normalize(self.q_proj_layer(hidden_states[:, 1:]), dim=-1),
                    F.normalize(self.k_proj_layer(hidden_states[:, :-1]), dim=-1),
                            )
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        # Force boundary probability of the first element to 1.0
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

        if self.training:
            sharp_boundary_prob = sharpen_prob_power(boundary_prob, self.bound_prob_sharp_rate)
            boundary_mask = torch.bernoulli(sharp_boundary_prob).bool()
        else:
            boundary_mask = boundary_prob >= 0.5

        # Each router's aux loss uses the union of its assigned tokens over views.
        if aux_token_mask is None:
            aux_token_mask = torch.ones_like(boundary_mask)
        if aux_token_mask.any():
            G = boundary_prob[aux_token_mask].mean()
            Freq = boundary_mask[aux_token_mask].float().mean()
            aux_loss = self.R/(self.R-1) * ((self.R-1)*Freq*G+(1-Freq)*(1-G))
        else:
            aux_loss = boundary_prob.sum() * 0
        selected_idx = boundary_mask.long()
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)


        selected_probs = boundary_prob.gather(
                    dim=-1, index=selected_idx.unsqueeze(-1)
                )  # (shape hidden_states.shape[:-1], 1)
        return boundary_prob, boundary_mask, selected_probs, aux_loss

def get_seq_idx(cu_seqlens, device=None):
    seq_idx = torch.zeros(cu_seqlens[-1], dtype=torch.long, device=device)
    seq_idx[cu_seqlens[:-1]] = 1
    seq_idx = (torch.cumsum(seq_idx, dim=0) - 1).unsqueeze(0).int()

    return seq_idx

class DechunkModule(nn.Module):
    def __init__(self, block_size=256, headdim=32, hidden_size=2048):
        super().__init__()
        self.block_size = block_size
        self.headdim = headdim
        assert hidden_size % self.headdim == 0
        self.nheads = hidden_size // self.headdim
        self.hidden_size = hidden_size
        self.dtype = torch.float32

    def forward(self, concept, selected_probs, boundary_mask):
        if mamba_chunk_scan_combined is None or not concept.is_cuda:
            concept_prob = selected_probs[boundary_mask].unsqueeze(0)     # shape [1,522,1]
            # For clarity, use a loop; the GPU path below uses a parallel scan.
            concept_merge = [concept[:, 0]]
            for i in range(1, concept.shape[1]):
                concept_merge.append(concept_merge[-1]*(1-concept_prob[:, i]) + concept[:, i]*concept_prob[:, i])
            concept_merge = torch.stack(concept_merge, dim=1)
        else:
            cu_seqlens = torch.tensor([0, concept.shape[1]], device=concept.device)
            seq_idx = get_seq_idx(cu_seqlens, device=concept.device)
            p = selected_probs[boundary_mask].unsqueeze(0).squeeze(-1).clamp(1e-4, 1 - 1e-4)
            original_dtype = concept.dtype
            # Reuse Mamba2 kernel for EMA Deaggregator.
            dt = torch.log(1 / (1 - p)).to(self.dtype)

            x = (concept / dt[..., None]).to(self.dtype)

            A = -torch.ones(
                (self.nheads,), device=concept.device, dtype=torch.float32
            )
            b = p.to(self.dtype)
            c = torch.ones_like(b)

            out = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                repeat(dt, "b l -> b l h", h=self.nheads),
                A,
                rearrange(b, "b l -> b l 1 1"),
                rearrange(c, "b l -> b l 1 1"),
                chunk_size=self.block_size,
                seq_idx=seq_idx,
            )
            concept_merge = rearrange(out, "b l h p -> b l (h p)")

        concept_merge = concept_merge.squeeze(0)
        plug_back_idx = boundary_mask.squeeze(0).cumsum(dim=0) - 1
        concept_merge = torch.gather(
            concept_merge, dim=0, index=plug_back_idx.unsqueeze(-1).expand(-1, self.hidden_size)
        )
        concept_merge = concept_merge.unsqueeze(0)
        return concept_merge

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)
    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        return grad_x
def ste_func(x):
    return STE.apply(x)

class MoELayer(nn.Module):
    def __init__(self, layer_id, use_joint_decoding=False):
        super().__init__()
        self.layer_id = layer_id
        self.use_joint_decoding = use_joint_decoding

        self.attn = lambda x:x
        self.moe  = lambda x:x
        self.attn_joint = lambda x,y:x

    def forward(self, hidden_states, concept_merge=None):
        if self.use_joint_decoding:
            hidden_states = hidden_states + self.attn_joint(hidden_states, concept_merge)
        else:
            hidden_states = hidden_states + self.attn(hidden_states)
        hidden_states += self.moe(hidden_states)
        return hidden_states

class ConceptMoE(nn.Module):
    def __init__(self, hidden_size=2048, vocab_size=32000, compress_ratio_list=(2, 4, 8)):
        super().__init__()

        self.encoder = nn.ModuleList(MoELayer(layer_id=i) for i in range(2))
        self.concept_model = nn.ModuleList(MoELayer(layer_id=i) for i in range(2,25))
        self.decoder = nn.ModuleList(MoELayer(layer_id=i, use_joint_decoding=True) for i in range(25,27))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Only the chunk routers are ratio-specific; all other modules are shared.
        self.compress_ratio_list = list(compress_ratio_list)
        self.chunk_module = nn.ModuleList(ChunkModule(hidden_size, r) for r in compress_ratio_list)
        self.dechunk_module = DechunkModule(hidden_size=hidden_size)

    def forward(self, input_ids, mixed_cr=False, compress_ratio=None):
        # Like the original demo, use one input sequence: [1, sequence_length].
        assert input_ids.shape[0] == 1
        assert not (mixed_cr and compress_ratio is not None)
        # encoder: run once for all compression views
        hidden_state = self.embedding(input_ids)
        for layer in self.encoder:
            hidden_state = layer(hidden_state)

        # chunk: fixed-ratio views, or independently sampled mixed-ratio views
        length = input_ids.shape[1]
        device = input_ids.device
        active = list(range(len(self.compress_ratio_list)))
        if compress_ratio is not None:
            active = [self.compress_ratio_list.index(compress_ratio)]
        ratio_ids = torch.tensor(active, device=device)[:, None].expand(-1, length).clone()
        segment_starts = torch.zeros_like(ratio_ids, dtype=torch.bool)
        segment_starts[:, 0] = True
        if mixed_cr:
            for view in range(len(active)):
                start = 0
                while start < length:
                    end = min(start + torch.randint(8, 25, (1,)).item(), length)
                    ratio_ids[view, start:end] = torch.randint(len(self.compress_ratio_list), (1,)).item()
                    segment_starts[view, start] = True
                    start = end

        router_outputs, aux_losses = [], []
        for i, router in enumerate(self.chunk_module):
            assigned_tokens = (ratio_ids == i).any(dim=0, keepdim=True)
            router_output = router(hidden_state, aux_token_mask=assigned_tokens)
            router_outputs.append(router_output)
            aux_losses.append(router_output[-1])
        all_probs = torch.cat([out[0] for out in router_outputs], dim=0)
        all_masks = torch.cat([out[1] for out in router_outputs], dim=0)

        logits = []
        for ids, starts in zip(ratio_ids, segment_starts):
            positions = torch.arange(length, device=device)
            boundary_prob = all_probs[ids, positions].unsqueeze(0)
            boundary_mask = all_masks[ids, positions].unsqueeze(0) | starts.unsqueeze(0)
            boundary_prob = torch.where(starts[None, :, None], boundary_prob.new_tensor([0., 1.]), boundary_prob)
            selected_probs = boundary_prob.gather(-1, boundary_mask.long().unsqueeze(-1))

            # main network: reuse the same weights for every view
            concept = hidden_state[boundary_mask].unsqueeze(0)
            for layer in self.concept_model:
                concept = layer(concept)

            # dechunk
            concept_merge = self.dechunk_module(concept, selected_probs, boundary_mask)

            # decoder
            decoded = hidden_state + concept_merge * ste_func(selected_probs)
            for layer in self.decoder:
                decoded = layer(decoded, concept_merge)  # joint decoding
            logits.append(self.lm_head(decoded))

        return torch.cat(logits, dim=0), sum(aux_losses)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConceptMoE(hidden_size=64, vocab_size=128).to(device)
    input_ids = torch.randint(0, 128, (1, 64), device=device)
    logits, aux_loss = model(input_ids)                 # one view per ratio: 2, 4, 8
    print(logits.shape, aux_loss)
    logits, aux_loss = model(input_ids, mixed_cr=True)  # three mixed-ratio views
    print(logits.shape, aux_loss)
