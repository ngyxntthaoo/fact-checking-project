"""
ces_module.py — Concept Evidence Selection (CES) with 01-GATE.

From Concept-HGN paper (Section 3.2):

  Given a sentence embedding h and concept embeddings {c_1, ..., c_m}:
  1. Compute similarity: s_j = dot(h, c_j)
  2. Normalize: p = softmax(s)
  3. 01-GATE filter:
       - zero out p_j where p_j < threshold alpha
       - keep top-k remaining
  4. Integrate: h_bar = h + sum_j(p_j * c_j)  [weighted sum over filtered concepts]

Hyperparams (from paper):
  k = 10   (top-k concepts)
  alpha = 0.8  (FEVER), 0.7 (UKP Snopes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroOneGate(nn.Module):
    """
    01-GATE: binary mask that zeroes out concepts below threshold alpha,
    then keeps only top-k among the remaining.
    Returns filtered softmax scores.
    """

    def __init__(self, alpha: float = 0.8, top_k: int = 10):
        super().__init__()
        self.alpha = alpha
        self.top_k = top_k

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: [num_concepts]  softmax probabilities
        Returns:
            filtered: [num_concepts]  re-normalized scores with low-score concepts zeroed out
        """
        # Step 1: zero out below threshold
        mask = (scores >= self.alpha).float()
        filtered = scores * mask

        # Step 2: keep top-k (if fewer than k survived the threshold, keep all survivors)
        if filtered.sum() > 0:
            k = min(self.top_k, (filtered > 0).sum().item())
            if k > 0:
                topk_vals, topk_idx = torch.topk(filtered, int(k))
                result = torch.zeros_like(filtered)
                result.scatter_(0, topk_idx, topk_vals)
                filtered = result

        # Re-normalize so weights sum to 1 (avoid zero division)
        total = filtered.sum()
        if total > 0:
            filtered = filtered / total

        return filtered


class CESModule(nn.Module):
    """
    Concept Evidence Selection module.

    Takes a sentence representation and a variable set of concept embeddings,
    filters via 01-GATE, then returns concept-enriched sentence representation.
    """

    def __init__(self, hidden_dim: int, alpha: float = 0.8, top_k: int = 10):
        super().__init__()
        self.gate = ZeroOneGate(alpha=alpha, top_k=top_k)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        sent_emb: torch.Tensor,
        concept_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sent_emb:     [d]              sentence/claim embedding (CLS token)
            concept_embs: [num_concepts, d] concept embeddings (also CLS of concept text)

        Returns:
            enriched: [d]  sent_emb enhanced with relevant concepts
        """
        if concept_embs.size(0) == 0:
            return sent_emb

        # Cosine similarity → softmax
        sent_norm = F.normalize(sent_emb.unsqueeze(0), dim=-1)       # [1, d]
        conc_norm = F.normalize(concept_embs, dim=-1)                 # [num_concepts, d]
        sim = (sent_norm @ conc_norm.T).squeeze(0)                    # [num_concepts]
        scores = F.softmax(sim, dim=0)

        # 01-GATE filter
        filtered_scores = self.gate(scores)                           # [num_concepts]

        # Weighted sum of concept embeddings
        concept_summary = (filtered_scores.unsqueeze(-1) * concept_embs).sum(0)  # [d]

        # Integrate: enriched = sent_emb + concept_summary
        enriched = sent_emb + concept_summary
        return enriched

    def forward_batch(
        self,
        sent_embs: torch.Tensor,
        concept_embs_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Batched forward for multiple sentences with different concept sets.

        Args:
            sent_embs:          [N, d]  N sentence embeddings
            concept_embs_list:  list of N tensors, each [num_concepts_i, d]

        Returns:
            enriched: [N, d]
        """
        enriched = []
        for i in range(sent_embs.size(0)):
            e = self.forward(sent_embs[i], concept_embs_list[i])
            enriched.append(e.unsqueeze(0))
        return torch.cat(enriched, dim=0)


class ConceptEncoder(nn.Module):
    """
    Encodes concept label strings into embeddings via the shared PLM.
    Concept text format: "<entity> is a <concept_label>"
    e.g. "Tesla is a company"
    """

    def __init__(self, plm, tokenizer, max_length: int = 32):
        super().__init__()
        self.plm = plm
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode_concepts(self, entity: str, concept_labels: list[str]) -> torch.Tensor:
        """
        Args:
            entity: surface entity text, e.g. "Tesla"
            concept_labels: list of concept label strings, e.g. ["company", "automaker"]

        Returns:
            concept_embs: [num_concepts, d]  CLS embeddings for each concept
        """
        if not concept_labels:
            return torch.zeros(0, self.plm.config.hidden_size,
                               device=next(self.plm.parameters()).device)

        texts = [f"{entity} is a {c}" for c in concept_labels]
        device = next(self.plm.parameters()).device
        enc = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        ).to(device)

        with torch.no_grad():
            out = self.plm(**enc, output_hidden_states=True)
        cls_embs = out.hidden_states[-1][:, 0, :]  # [num_concepts, d]
        return cls_embs
