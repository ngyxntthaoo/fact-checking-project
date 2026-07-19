"""
concept_hgn_model.py — Concept-HGN model implementation.

Implements the 5 modules from the paper:
  1. BERT Encoding       — 3 input types: BERT(c), BERT(c,vi), BERT(vi)
  2. CES (01-GATE)       — concept-enriched representations
  3. Multi-Level Interaction — word-level MHSA + sentence-level MHSA
  4. Graph Construction + GAT — edge-type-aware R-GAT, 4 relations
  5. Evidence Aggregation — Gate Attention + MLP classifier

References HeterFC model.py for:
  - RGATConv import (already in HeterFC's model.py)
  - nn.ModuleList pattern for GNN layers
  - forward() signature style

Node layout per sample (from preprocess_fever.py):
  [0]       = claim node
  [1..N]    = evidence nodes
  [N+1..]   = entity nodes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# RGATConv is already imported in HeterFC's model.py — reuse same import
from torch_geometric.nn import RGATConv

from ces_module import CESModule, ConceptEncoder


class MultiLevelInteraction(nn.Module):
    """
    Module 3: Multi-Level Interaction via MHSA.

    Word-level: apply MHSA over all token hidden states of all evidences.
      H̃ = H̄ + MHSA(H̄)

    Sentence-level: apply MHSA over the CLS tokens of each evidence.
      F = stack of CLS tokens from H̃
      S = F + MHSA(F)

    Returns S: [N, d] sentence-level textual representation.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.word_mhsa = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.sent_mhsa = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, evidence_token_hiddens: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            evidence_token_hiddens: list of N tensors, each [seq_len, d]
                                    token-level hidden states for each evidence

        Returns:
            S: [N, d]  sentence-level representations after MHSA
        """
        # Word-level MHSA: process each evidence sentence independently
        cls_tokens = []
        for tok_hidden in evidence_token_hiddens:
            h = tok_hidden.unsqueeze(0)                     # [1, seq_len, d]
            attn_out, _ = self.word_mhsa(h, h, h)          # [1, seq_len, d]
            h_tilde = self.norm1(h + attn_out)              # residual + norm
            cls_tokens.append(h_tilde[0, 0, :])            # CLS token = [d]

        # Sentence-level MHSA: over stacked CLS tokens
        F_mat = torch.stack(cls_tokens, dim=0).unsqueeze(0)  # [1, N, d]
        attn_out, _ = self.sent_mhsa(F_mat, F_mat, F_mat)   # [1, N, d]
        S = self.norm2(F_mat + attn_out).squeeze(0)          # [N, d]
        return S


class GateAttentionAggregator(nn.Module):
    """
    Module 5: Evidence Aggregation via Gate Attention.

    From paper description:
      C = ReLU(W_c · [S, M])    — transform combined textual+graph rep
      M̄ = softmax(C) ⊙ M        — gate selects relevant graph features
      gate = Sigmoid(W_g · [S, M̄])
      Q = gate ⊙ Tanh(M̄)       — final aggregated representation

    Then MLP + Softmax → 3-class prediction.
    """

    def __init__(self, hidden_dim: int, num_class: int = 3, dropout: float = 0.5):
        super().__init__()
        self.W_c = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W_g = nn.Linear(2 * hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, S: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S: [N, d]  sentence-level textual representations (Module 3 output)
            M: [N, d]  graph representations for evidence nodes (Module 4 output)

        Returns:
            logits: [d_class]  unnormalized class scores
        """
        # C: transform combined representation
        SM = torch.cat([S, M], dim=-1)              # [N, 2d]
        C = F.relu(self.W_c(SM))                    # [N, d]

        # M̄: gated graph representation
        attn_weights = F.softmax(C, dim=0)          # [N, d]  element-wise softmax over N
        M_bar = attn_weights * M                    # [N, d]

        # Gate
        gate_input = torch.cat([S, M_bar], dim=-1) # [N, 2d]
        gate = torch.sigmoid(self.W_g(gate_input)) # [N, d]
        Q = gate * torch.tanh(M_bar)               # [N, d]

        # Aggregate over N evidences: mean pool
        Q_agg = Q.mean(dim=0)                      # [d]
        Q_agg = self.dropout(Q_agg)
        logits = self.classifier(Q_agg)            # [num_class]
        return logits


class ConceptHGN(nn.Module):
    """
    Concept-enhanced Heterogeneous Graph Network for Fact Verification.

    Args:
        plm:         HuggingFace RoBERTa (or compatible) model
        tokenizer:   corresponding tokenizer (for ConceptEncoder)
        hidden_dim:  PLM hidden size (1024 for RoBERTa-Large)
        num_heads:   MHSA heads (paper: 8)
        num_relations: 4 edge types in heterogeneous graph
        num_class:   3 (SUPPORTS / REFUTES / NEI)
        gnn_layers:  number of R-GAT layers (paper: 2)
        alpha:       CES 01-GATE threshold (0.8 for FEVER)
        top_k:       CES top-k concepts (10)
        dropout:     dropout rate
    """

    def __init__(
        self,
        plm,
        tokenizer,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_relations: int = 4,
        num_class: int = 3,
        gnn_layers: int = 2,
        alpha: float = 0.8,
        top_k: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.plm = plm
        self.hidden_dim = hidden_dim

        # Module 2: CES + ConceptEncoder
        self.concept_encoder = ConceptEncoder(plm, tokenizer)
        self.ces = CESModule(hidden_dim, alpha=alpha, top_k=top_k)

        # Module 3: Multi-Level Interaction
        self.multi_level = MultiLevelInteraction(hidden_dim, num_heads, dropout)

        # Module 4: Heterogeneous Graph — R-GAT (already imported from HeterFC model.py)
        # nn.ModuleList pattern reused from HeterFC
        gnn_list = []
        for layer in range(gnn_layers):
            gnn_list.append(RGATConv(hidden_dim, hidden_dim, num_relations=num_relations))
        self.gnns = nn.ModuleList(gnn_list)

        # Module 5: Gate Attention + classifier
        self.aggregator = GateAttentionAggregator(hidden_dim, num_class, dropout)

        self.dropout = nn.Dropout(dropout)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run PLM and return last hidden states: [batch, seq_len, d]."""
        out = self.plm(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return out.hidden_states[-1]

    def forward(
        self,
        batch,
        concept_data: list[dict] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            batch: PyG Batch object with fields from preprocess_fever.pack_graph_data()
            concept_data: optional list of per-sample concept dicts
                          (from concept_extractor.get_sample_concepts)
                          If None, CES is skipped (ablation: w/o concept).

        Returns:
            logits: [batch_size, num_class]
        """
        # Handle both batched and single-sample inputs
        device = next(self.parameters()).device
        if hasattr(batch, 'to_data_list'):
            data_list = batch.to_data_list()
        else:
            data_list = [batch]

        all_logits = []

        # Process each sample independently (variable evidence counts)
        # For batching we use accumulation — same approach as HeterFC train.py
        for i, data in enumerate(data_list):
            logit = self._forward_single(data, i, concept_data, device)
            all_logits.append(logit.unsqueeze(0))

        return torch.cat(all_logits, dim=0)

    def _forward_single(self, data, idx: int, concept_data, device) -> torch.Tensor:
        N = data.num_evidence

        # ── Module 1: BERT Encoding ──────────────────────────────────────────
        # h_c: claim CLS  [d]
        h_c_all = self.encode(
            data.claim_ids.to(device),
            data.claim_mask.to(device)
        )
        h_c = h_c_all[0, 0, :]  # CLS token

        # h_i: claim-aware evidence CLS [N, d]
        h_i_all = self.encode(
            data.ce_ids.to(device),
            data.ce_mask.to(device)
        )
        h_i = h_i_all[:N, 0, :]  # [N, d]

        # h_vi: evidence token hidden states [N, seq_len, d]
        h_vi_all = self.encode(
            data.e_ids.to(device),
            data.e_mask.to(device)
        )
        h_vi = [h_vi_all[j] for j in range(N)]  # list of [seq_len, d]

        # ── Module 2: CES (Concept Evidence Selection) ───────────────────────
        if concept_data is not None:
            cd = concept_data[idx]
            # Enrich claim representation
            claim_concept_embs = self._encode_concepts(
                cd['claim_entities'], cd['claim_concepts']
            )
            h_c_bar = self.ces(h_c, claim_concept_embs)

            # Enrich each evidence representation
            h_i_bar_list = []
            for j in range(N):
                ent_list = cd['evi_entities'][j] if j < len(cd['evi_entities']) else []
                conc_map = cd['evi_concepts'][j] if j < len(cd['evi_concepts']) else {}
                evi_concept_embs = self._encode_concepts(ent_list, conc_map)
                h_i_bar_list.append(self.ces(h_i[j], evi_concept_embs))
            h_i_bar = torch.stack(h_i_bar_list, dim=0)  # [N, d]
        else:
            h_c_bar = h_c
            h_i_bar = h_i

        # Inject enriched representations back into token hiddens at CLS position
        for j in range(N):
            h_vi[j] = h_vi[j].clone()
            h_vi[j][0] = h_i_bar[j]  # replace CLS with enriched representation

        # ── Module 3: Multi-Level Interaction ────────────────────────────────
        S = self.multi_level(h_vi)  # [N, d]

        # ── Module 4: Heterogeneous Graph GAT ────────────────────────────────
        # Build initial node feature matrix X
        # Node 0: claim   Nodes 1..N: evidences
        # Entity nodes get zero init (no separate encoding in basic version)
        edge_index = data.edge_index.to(device)
        edge_type = data.edge_type.to(device)

        # Count total nodes = 1 (claim) + N (evidence) + E (entities)
        total_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else N + 1
        E = total_nodes - (N + 1)

        X = torch.zeros(total_nodes, self.hidden_dim, device=device)
        X[0] = h_c_bar                              # claim node
        X[1:N+1] = h_i_bar                          # evidence nodes

        # R-GAT propagation (nn.ModuleList pattern from HeterFC model.py)
        for gnn in self.gnns:
            X = F.relu(gnn(X, edge_index, edge_type))

        M = X[1:N+1]  # [N, d]  evidence node representations after GAT

        # ── Module 5: Evidence Aggregation + Prediction ──────────────────────
        logit = self.aggregator(S, M)  # [num_class]
        return logit

    def _encode_concepts(self, entity_list: list[str], concept_map: dict) -> torch.Tensor:
        """Encode all concepts for given entities, return [total_concepts, d]."""
        all_embs = []
        for entity in entity_list:
            concept_labels = concept_map.get(entity, [])
            if concept_labels:
                embs = self.concept_encoder.encode_concepts(entity, concept_labels)
                all_embs.append(embs)
        if all_embs:
            return torch.cat(all_embs, dim=0)
        device = next(self.parameters()).device
        return torch.zeros(0, self.hidden_dim, device=device)
