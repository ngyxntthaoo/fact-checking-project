"""
preprocess_fever.py — Preprocessing pipeline for Concept-HGN on FEVER.

Reuses from HeterFC:
  - myset Dataset class (copied verbatim)
  - pickle cache pattern
  - pack_graph_data() adapted from prepare_graph_data_word_level_attention()

Rewrites:
  - prepare_plm_input()  : 3 input types per paper (BERT(c), BERT(c,vi), BERT(vi))
  - build_sentence_graph(): sentence-level nodes, 4 edge types

Graph structure:
  Node types:   0=claim, 1=evidence, 2=entity
  Edge types:   0=claim-evidence, 1=claim-entity, 2=evidence-entity, 3=evidence-evidence
  Node index layout (per sample):
    [0]         : claim node
    [1..N]      : evidence nodes  (N = num evidences)
    [N+1..N+E]  : entity nodes    (E = total unique entities across claim+evidences)
"""

import os
import os.path as osp
import pickle
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data


# ── copied verbatim from HeterFC preprocess.py ──────────────────────────────
class myset(Dataset):
    def __init__(self, datalist):
        self.x = datalist

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)
# ────────────────────────────────────────────────────────────────────────────


MAX_EVIDENCE = 5
MAX_SEQ_LEN = 128
MAX_CONCAT_LEN = 512


def prepare_plm_input(samples: list[dict], tokenizer, max_length: int = MAX_SEQ_LEN) -> list[dict]:
    """
    For each sample encode 3 input types required by Concept-HGN:
      - enc_c      : BERT(claim)               → CLS = claim representation h_c
      - enc_ce     : BERT(claim </s> vi) × N   → CLS = claim-aware evidence h_i
      - enc_e      : BERT(vi) × N              → all token hidden states = h_vi

    Returns list of dicts with tokenized tensors + metadata.
    """
    results = []
    sep = '</s></s>'  # RoBERTa separator

    for sample in tqdm(samples, desc='Tokenizing'):
        claim = sample['claim']
        evidences = sample['evidence_texts'][:MAX_EVIDENCE]
        label = sample['label']
        N = len(evidences)

        # 1. Encode claim only
        enc_c = tokenizer(
            claim, padding='max_length', truncation=True,
            max_length=max_length, return_tensors='pt'
        )

        # 2. Encode claim + each evidence (claim-aware evidence representation)
        claim_evi_texts = [claim + sep + e for e in evidences]
        enc_ce = tokenizer(
            claim_evi_texts, padding='max_length', truncation=True,
            max_length=max_length, return_tensors='pt'
        )

        # 3. Encode each evidence alone (for word-level token hidden states in MHSA)
        enc_e = tokenizer(
            evidences, padding='max_length', truncation=True,
            max_length=max_length, return_tensors='pt'
        )

        # 4. Full concat for fused prediction branch (HeterFC pattern)
        full_text = claim + ' ' + ' </s> '.join(evidences)
        enc_full = tokenizer(
            full_text, padding='max_length', truncation=True,
            max_length=MAX_CONCAT_LEN, return_tensors='pt'
        )

        results.append({
            'id': sample.get('id', -1),
            'label': label,
            'num_evidence': N,
            'enc_c': enc_c,         # claim only
            'enc_ce': enc_ce,       # [N, seq_len]  claim + evidence
            'enc_e': enc_e,         # [N, seq_len]  evidence only
            'enc_full': enc_full,   # [1, 512]      full concat
            # entity info filled in later by build_sentence_graph
            'claim_entities': sample.get('claim_entities', []),
            'evi_entities': sample.get('evi_entities', [[] for _ in evidences]),
        })

    return results


def build_sentence_graph(preprocessed: list[dict]) -> tuple[list, list]:
    """
    Build a sentence-level heterogeneous graph per sample.

    Node layout:
      [0]      = claim node
      [1..N]   = evidence nodes
      [N+1..]  = entity nodes (unique entities across claim + all evidences)

    Edge types:
      0 = claim–evidence   (bidirectional)
      1 = claim–entity     (bidirectional)
      2 = evidence–entity  (bidirectional)
      3 = evidence–evidence (fully connected, bidirectional)

    Returns:
      edge_index_list: list of np.array [2, num_edges]
      edge_type_list:  list of np.array [num_edges]
    """
    edge_index_list = []
    edge_type_list = []

    for item in tqdm(preprocessed, desc='Building graphs'):
        N = item['num_evidence']           # number of evidence nodes
        claim_entities = item['claim_entities']
        evi_entities = item['evi_entities']

        # Collect unique entities, assign node indices starting from N+1
        all_entities = list(claim_entities)
        for ent_list in evi_entities:
            for e in ent_list:
                if e not in all_entities:
                    all_entities.append(e)

        E = len(all_entities)
        ent_to_idx = {e: N + 1 + i for i, e in enumerate(all_entities)}

        edges = []     # list of [src, dst] pairs
        etypes = []    # corresponding edge types

        def add_edge(u, v, etype):
            edges.append([u, v])
            edges.append([v, u])
            etypes.append(etype)
            etypes.append(etype)

        # Edge type 0: claim (node 0) ↔ each evidence node
        for i in range(1, N + 1):
            add_edge(0, i, 0)

        # Edge type 1: claim (node 0) ↔ entity nodes from claim
        for ent in claim_entities:
            if ent in ent_to_idx:
                add_edge(0, ent_to_idx[ent], 1)

        # Edge type 2: each evidence node ↔ its entity nodes
        for i, ent_list in enumerate(evi_entities):
            evi_node = i + 1
            for ent in ent_list:
                if ent in ent_to_idx:
                    add_edge(evi_node, ent_to_idx[ent], 2)

        # Edge type 3: evidence–evidence (fully connected)
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                add_edge(i, j, 3)

        if edges:
            edge_index = np.array(edges, dtype=np.int64).T
            edge_type = np.array(etypes, dtype=np.int64)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_type = np.zeros(0, dtype=np.int64)

        edge_index_list.append(edge_index)
        edge_type_list.append(edge_type)

    return edge_index_list, edge_type_list


def pack_graph_data(preprocessed: list[dict], edge_index_list: list, edge_type_list: list) -> list:
    """
    Pack everything into PyG Data objects.
    Adapted from HeterFC's prepare_graph_data_word_level_attention()
    — removed gold_indicator (not used in Concept-HGN loss).
    """
    data_list = []

    for idx, (item, edge_index, edge_type) in enumerate(
        tqdm(zip(preprocessed, edge_index_list, edge_type_list),
             desc='Packing PyG Data', total=len(preprocessed))
    ):
        N = item['num_evidence']

        # Input IDs: stack [claim_only, ce_1, ..., ce_N, e_1, ..., e_N]
        # Model will slice these apart in forward()
        claim_ids = item['enc_c']['input_ids']              # [1, seq]
        ce_ids = item['enc_ce']['input_ids']                # [N, seq]
        e_ids = item['enc_e']['input_ids']                  # [N, seq]
        full_ids = item['enc_full']['input_ids']            # [1, 512]

        claim_mask = item['enc_c']['attention_mask']
        ce_mask = item['enc_ce']['attention_mask']
        e_mask = item['enc_e']['attention_mask']
        full_mask = item['enc_full']['attention_mask']

        total_nodes = int(edge_index.max()) + 1 if edge_index.size > 0 else (N + 1)

        data = Data(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_type=torch.tensor(edge_type, dtype=torch.long),
            y=torch.tensor(item['label'], dtype=torch.long),
            num_nodes=total_nodes,
        )
        data.claim_ids = claim_ids          # [1, seq]
        data.claim_mask = claim_mask
        data.ce_ids = ce_ids                # [N, seq]
        data.ce_mask = ce_mask
        data.e_ids = e_ids                  # [N, seq]
        data.e_mask = e_mask
        data.full_ids = full_ids            # [1, 512]
        data.full_mask = full_mask
        data.num_evidence = N
        data.sample_id = item['id']
        data.sample_idx = idx          # positional index for concept_data lookup

        data_list.append(data)

    return data_list


def preprocess_fever(
    samples: list[dict],
    tokenizer,
    split: str,
    cache_dir: str = 'preprocessed',
    max_length: int = MAX_SEQ_LEN,
) -> list:
    """
    Full preprocessing pipeline with pickle caching (HeterFC pattern).

    Args:
        samples: output of fever_loader.load_fever() — must include
                 'claim_entities' and 'evi_entities' fields added by concept_extractor
        tokenizer: HuggingFace tokenizer
        split: 'train' | 'dev' | 'test'
        cache_dir: directory for .pkl files
        max_length: max token sequence length

    Returns:
        list of PyG Data objects
    """
    os.makedirs(cache_dir, exist_ok=True)
    plm_cache = osp.join(cache_dir, f'{split}_plm.pkl')
    graph_cache = osp.join(cache_dir, f'{split}_graph.pkl')

    # ── PLM tokenization (cached) ────────────────────────────────────────────
    if osp.exists(plm_cache):
        print(f'Loading cached PLM inputs from {plm_cache}')
        with open(plm_cache, 'rb') as f:
            preprocessed = pickle.load(f)
    else:
        preprocessed = prepare_plm_input(samples, tokenizer, max_length)
        with open(plm_cache, 'wb') as f:
            pickle.dump(preprocessed, f)

    # ── Graph construction (cached) ──────────────────────────────────────────
    if osp.exists(graph_cache):
        print(f'Loading cached graphs from {graph_cache}')
        with open(graph_cache, 'rb') as f:
            edge_index_list, edge_type_list = pickle.load(f)
    else:
        edge_index_list, edge_type_list = build_sentence_graph(preprocessed)
        with open(graph_cache, 'wb') as f:
            pickle.dump((edge_index_list, edge_type_list), f)

    return pack_graph_data(preprocessed, edge_index_list, edge_type_list)
