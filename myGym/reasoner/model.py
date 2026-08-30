"""
Reasoner Module — multimodal reasoning for object grounding and action sequencing.

Architecture overview:
  1. Encoders: HumanPoseEncoder (GCN), ObjectSetEncoder (MLP+Transformer),
     LanguageEncoder (Transformer)
  2. Grounding: three-path direct + context + pose scoring
  3. Action: Cross-attention fusion → autoregressive decoder

Grounding uses a direct raw-embedding match as its primary signal:
  color_emb · raw_lang_emb is maximised for the object whose color matches
  a language token, because both use the *same* shared embedding table.
  This gives a strong non-random signal from initialisation, before any
  MLP or transformer changes the representations.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.1  Human Pose Encoder — Graph Convolutional Network
# ---------------------------------------------------------------------------

class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(x, self.weight)
        return torch.matmul(adj, support) + self.bias


class HumanPoseEncoder(nn.Module):
    """GCN over skeleton joints; max-pools across joints to preserve pointing signal."""

    def __init__(
        self,
        num_joints: int = 18,
        coord_dim: int = 3,
        hidden_dim: int = 64,
        out_dim: int = 128,
        adjacency: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_joints = num_joints
        if adjacency is None:
            adjacency = torch.ones(num_joints, num_joints)
        self.register_buffer("adj", self._normalise_adjacency(adjacency))
        self.gc1  = GraphConvolution(coord_dim, hidden_dim)
        self.gc2  = GraphConvolution(hidden_dim, hidden_dim)
        self.gc3  = GraphConvolution(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    @staticmethod
    def _normalise_adjacency(adj: torch.Tensor) -> torch.Tensor:
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj.sum(dim=1).clamp(min=1)
        d   = deg.pow(-0.5)
        return d.unsqueeze(1) * adj * d.unsqueeze(0)

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x, self.adj))
        x = F.relu(self.gc2(x, self.adj))
        x = self.gc3(x, self.adj)
        return x.max(dim=1).values          # max-pool over joints

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        if pose.dim() == 4:
            B, T, K, C = pose.shape
            enc = self._encode_frames(pose.view(B * T, K, C))
            x   = enc.view(B, T, -1).max(dim=1).values
        else:
            x = self._encode_frames(pose)
        return self.norm(x)


# ---------------------------------------------------------------------------
# 1.2  Object Set Encoder
# ---------------------------------------------------------------------------

class ObjectSetEncoder(nn.Module):
    """
    Encodes N objects into contextualised per-object embeddings and also
    returns the **raw** color / shape embeddings before the MLP projection.

    The raw embeddings are needed for the direct-match grounding path: they
    share weights with the language embedding table so that "green" as an
    object colour and "green" as a language token start as the same vector.

    Input tensors:
        objects  (B, N, 6)  — continuous [x, y, z, roll, pitch, yaw]
        obj_cat  (B, N, 2)  — int [color_id, shape_id] in language vocab space

    Returns:
        obj_emb   (B, N, out_dim)  — contextualised object embeddings
        color_emb (B, N, D)        — raw colour embedding (pre-MLP)
        shape_emb (B, N, D)        — raw shape  embedding (pre-MLP)
    """

    CONT_DIM = 6

    def __init__(
        self,
        lang_vocab_size: int,
        cat_embed_dim: int = 32,
        hidden_dim: int = 128,
        out_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_objects: int = 32,
    ):
        super().__init__()
        self.cat_embed = nn.Embedding(lang_vocab_size, cat_embed_dim)

        input_dim = self.CONT_DIM + 2 * cat_embed_dim
        self.obj_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),   nn.ReLU(),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        objects: torch.Tensor,
        obj_cat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        color_emb = self.cat_embed(obj_cat[:, :, 0])   # (B, N, cat_embed_dim)
        shape_emb = self.cat_embed(obj_cat[:, :, 1])   # (B, N, cat_embed_dim)
        x = torch.cat([objects, color_emb, shape_emb], dim=-1)
        x = self.obj_mlp(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.norm(x), color_emb, shape_emb


# ---------------------------------------------------------------------------
# 1.3  Language Encoder
# ---------------------------------------------------------------------------

class LanguageEncoder(nn.Module):
    """
    Returns seq_out (post-transformer), sent_vec (mean-pool), and raw_emb
    (pre-transformer token embeddings).

    raw_emb shares weights with ObjectSetEncoder.cat_embed when cat_embed_dim
    equals embed_dim, giving the direct-match grounding path its strong signal.
    """

    def __init__(
        self,
        vocab_size: int = 0,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim) if vocab_size > 0 else None
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            seq_out:  (B, T, D) — post-transformer per-token vectors
            sent_vec: (B, D)    — mean-pooled sentence vector
            raw_emb:  (B, T, D) — pre-transformer token embeddings
        """
        if tokens.dtype in (torch.long, torch.int):
            raw_emb = self.embedding(tokens)
        else:
            raw_emb = tokens
        T = raw_emb.size(1)
        x = raw_emb + self.pos_encoding[:, :T, :]
        seq_out = self.transformer(x, src_key_padding_mask=mask)
        seq_out = self.norm(seq_out)

        if mask is not None:
            lengths  = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
            sent_vec = (seq_out * (~mask).unsqueeze(-1).float()).sum(1) / lengths
        else:
            sent_vec = seq_out.mean(dim=1)

        return seq_out, sent_vec, raw_emb


# ---------------------------------------------------------------------------
# 2.  Cross-Attention Fusion (used for action context only)
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.query_proj = nn.Linear(d_model * 2, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        sent_vec: torch.Tensor,
        pose_vec: torch.Tensor,
        obj_embeddings: torch.Tensor,
        obj_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query   = self.query_proj(torch.cat([sent_vec, pose_vec], dim=-1)).unsqueeze(1)
        out, _  = self.cross_attn(query, obj_embeddings, obj_embeddings, key_padding_mask=obj_mask)
        return self.norm(out.squeeze(1))


# ---------------------------------------------------------------------------
# 3.1  Object Grounding Head — three-path scorer
# ---------------------------------------------------------------------------

class ObjectGroundingHead(nn.Module):
    """
    Three complementary grounding signals, summed:

    1. **Direct raw match** (strongest from init, no training needed):
       For each object, compute max dot-product between its raw colour/shape
       embedding and every raw language token embedding.  Because both use the
       same shared embedding table, "green" object vs "green" token gives
       ||E["green"]||² immediately — no learned transformation required.

    2. **Context match**:
       Language tokens (post-transformer) attend to each object embedding
       (post-MLP).  Learns complex, contextualised alignments.

    3. **Pose match**:
       Dot-product between the pose vector and each object embedding.
       Encodes spatial pointing direction.

    Learnable scale parameters (α, β, γ) weight the three paths.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.pose_proj  = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.tensor(1.0))   # raw match weight
        self.beta  = nn.Parameter(torch.tensor(1.0))   # context match weight
        self.gamma = nn.Parameter(torch.tensor(0.5))   # pose weight

    def forward(
        self,
        raw_lang:  torch.Tensor,             # (B, T, D) raw pre-transformer lang embeddings
        lang_seq:  torch.Tensor,             # (B, T, D) post-transformer lang embeddings
        obj_emb:   torch.Tensor,             # (B, N, D) post-MLP object embeddings
        color_emb: torch.Tensor,             # (B, N, E) raw colour embeddings (pre-MLP)
        shape_emb: torch.Tensor,             # (B, N, E) raw shape  embeddings (pre-MLP)
        pose_vec:  torch.Tensor,             # (B, D)
        obj_mask:  Optional[torch.Tensor] = None,
        lang_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        D = obj_emb.size(-1)
        E = color_emb.size(-1)

        # 1. Direct raw match — leverages shared embedding weights
        #    Normalise by sqrt(dim) to keep magnitudes comparable
        raw_lang_c = raw_lang[:, :, :E]  # take first E dims if E < D
        c_sim = torch.bmm(color_emb, raw_lang_c.transpose(1, 2)) / math.sqrt(E)  # (B,N,T)
        s_sim = torch.bmm(shape_emb, raw_lang_c.transpose(1, 2)) / math.sqrt(E)  # (B,N,T)
        if lang_mask is not None:
            inf_mask = lang_mask.unsqueeze(1)          # (B,1,T)
            c_sim = c_sim.masked_fill(inf_mask, float('-inf'))
            s_sim = s_sim.masked_fill(inf_mask, float('-inf'))
        raw_score = self.alpha * (c_sim.max(dim=2).values + s_sim.max(dim=2).values)  # (B,N)

        # 2. Context match — post-transformer language vs post-MLP objects
        ctx = torch.bmm(lang_seq, obj_emb.transpose(1, 2)) / math.sqrt(D)  # (B,T,N)
        if lang_mask is not None:
            ctx = ctx.masked_fill(lang_mask.unsqueeze(2), float('-inf'))
        ctx_score = self.beta * ctx.max(dim=1).values                       # (B,N)

        # 3. Pose match — pointing direction
        pose_q      = self.pose_proj(pose_vec).unsqueeze(1)                 # (B,1,D)
        pose_score  = self.gamma * torch.bmm(pose_q, obj_emb.transpose(1, 2)).squeeze(1)  # (B,N)

        logits = raw_score + ctx_score + pose_score
        if obj_mask is not None:
            logits = logits.masked_fill(obj_mask, float('-inf'))
        return logits


# ---------------------------------------------------------------------------
# 3.2  Action Sequencer Head — Autoregressive Transformer Decoder
# ---------------------------------------------------------------------------

class ActionSequencerHead(nn.Module):
    def __init__(
        self,
        action_vocab_size: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 32,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model           = d_model
        self.pad_token_id      = pad_token_id
        self.max_seq_len       = max_seq_len
        self.action_vocab_size = action_vocab_size

        self.token_embed    = nn.Embedding(action_vocab_size, d_model)
        self.pos_encoding   = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        dec_layer           = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.output_proj    = nn.Linear(d_model, action_vocab_size)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(
        self,
        context: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S  = tgt_tokens.shape
        x     = self.token_embed(tgt_tokens) + self.pos_encoding[:, :S, :]
        memory = context.unsqueeze(1)
        causal = self._causal_mask(S, x.device)
        out   = self.transformer_decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=tgt_mask)
        return self.output_proj(out)

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        start_token_id: int = 1,
        end_token_id: int = 2,
        max_len: int = 20,
    ) -> torch.Tensor:
        B      = context.size(0)
        device = context.device
        memory = context.unsqueeze(1)
        tokens = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)
        for _ in range(min(max_len - 1, self.max_seq_len - 1)):
            S      = tokens.size(1)
            x      = self.token_embed(tokens) + self.pos_encoding[:, :S, :]
            causal = self._causal_mask(S, device)
            out    = self.transformer_decoder(x, memory, tgt_mask=causal)
            next_t = self.output_proj(out[:, -1, :]).argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_t], dim=1)
            if (next_t == end_token_id).all():
                break
        return tokens


# ---------------------------------------------------------------------------
# Full Reasoner Model
# ---------------------------------------------------------------------------

class ReasonerModel(nn.Module):
    """
    Multimodal reasoner: human pose + object set + language → grounding + action.

    Grounding uses a three-path scorer (direct / context / pose) so that
    colour/shape matches in the language are immediately effective even at
    random initialisation, via the shared embedding table.
    """

    def __init__(
        self,
        num_joints: int = 18,
        lang_vocab_size: int = 64,
        cat_embed_dim: int = 32,
        action_vocab_size: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        max_objects: int = 32,
        max_action_len: int = 32,
        max_seq_len: int = 64,
        pad_token_id: int = 0,
        adjacency: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.d_model = d_model

        self.pose_encoder = HumanPoseEncoder(
            num_joints=num_joints, coord_dim=3,
            hidden_dim=d_model, out_dim=d_model, adjacency=adjacency,
        )
        self.object_encoder = ObjectSetEncoder(
            lang_vocab_size=lang_vocab_size, cat_embed_dim=cat_embed_dim,
            hidden_dim=d_model, out_dim=d_model, n_heads=n_heads, max_objects=max_objects,
        )
        self.language_encoder = LanguageEncoder(
            vocab_size=lang_vocab_size, embed_dim=d_model,
            n_heads=n_heads, max_seq_len=max_seq_len,
        )

        # Share embedding weights when dimensions allow — "green" is then the
        # same vector in both the language encoder and the object colour encoder.
        if cat_embed_dim == d_model:
            self.object_encoder.cat_embed.weight = self.language_encoder.embedding.weight

        self.fusion       = CrossAttentionFusion(d_model=d_model, n_heads=n_heads)
        self.object_head  = ObjectGroundingHead(d_model=d_model)
        self.action_head  = ActionSequencerHead(
            action_vocab_size=action_vocab_size, d_model=d_model,
            n_heads=n_heads, max_seq_len=max_action_len, pad_token_id=pad_token_id,
        )

    def _encode(self, pose, objects, obj_cat, lang_tokens, obj_mask, lang_mask):
        pose_vec                         = self.pose_encoder(pose)
        obj_emb, color_emb, shape_emb   = self.object_encoder(objects, obj_cat, mask=obj_mask)
        seq_out, sent_vec, raw_emb       = self.language_encoder(lang_tokens, mask=lang_mask)
        return pose_vec, obj_emb, color_emb, shape_emb, seq_out, sent_vec, raw_emb

    def forward(
        self,
        pose: torch.Tensor,
        objects: torch.Tensor,
        obj_cat: torch.Tensor,
        lang_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        obj_mask: Optional[torch.Tensor] = None,
        lang_mask: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pose_vec, obj_emb, color_emb, shape_emb, seq_out, sent_vec, raw_emb = \
            self._encode(pose, objects, obj_cat, lang_tokens, obj_mask, lang_mask)

        # Grounding (three-path scorer)
        obj_logits = self.object_head(
            raw_emb, seq_out, obj_emb, color_emb, shape_emb,
            pose_vec, obj_mask=obj_mask, lang_mask=lang_mask,
        )

        # Action generation (cross-attention fusion → decoder)
        context       = self.fusion(sent_vec, pose_vec, obj_emb, obj_mask=obj_mask)
        action_logits = self.action_head(context, action_tokens, tgt_mask=action_mask)

        return {"obj_logits": obj_logits, "action_logits": action_logits, "context": context}

    @torch.no_grad()
    def predict(
        self,
        pose: torch.Tensor,
        objects: torch.Tensor,
        obj_cat: torch.Tensor,
        lang_tokens: torch.Tensor,
        obj_mask: Optional[torch.Tensor] = None,
        lang_mask: Optional[torch.Tensor] = None,
        start_token_id: int = 1,
        end_token_id: int = 2,
        max_action_len: int = 20,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        pose_vec, obj_emb, color_emb, shape_emb, seq_out, sent_vec, raw_emb = \
            self._encode(pose, objects, obj_cat, lang_tokens, obj_mask, lang_mask)

        obj_logits = self.object_head(
            raw_emb, seq_out, obj_emb, color_emb, shape_emb,
            pose_vec, obj_mask=obj_mask, lang_mask=lang_mask,
        )
        obj_idx = obj_logits.argmax(dim=-1)

        context    = self.fusion(sent_vec, pose_vec, obj_emb, obj_mask=obj_mask)
        action_seq = self.action_head.generate(
            context, start_token_id=start_token_id,
            end_token_id=end_token_id, max_len=max_action_len,
        )
        return {"obj_logits": obj_logits, "obj_idx": obj_idx,
                "action_seq": action_seq, "context": context}
