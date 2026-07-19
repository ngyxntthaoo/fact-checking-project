# Concept-HGN Re-implementation — Phân tích kiến trúc & bản đồ code

## 1. Luồng dữ liệu tổng thể

```
FEVER train.jsonl / dev.jsonl
        │
        ▼ fever_loader.py: load_fever()
  [{claim, label, evidence_texts: [str × ≤5]}]
        │
        ▼ concept_extractor.py: get_sample_concepts()
  Gắn thêm: claim_entities, evi_entities, concept_cache
        │
        ▼ preprocess_fever.py: preprocess_fever()
  ├── prepare_plm_input()   → enc_c, enc_ce, enc_e, enc_full
  ├── build_sentence_graph() → edge_index [2, E], edge_type [E]
  └── pack_graph_data()     → list[PyG Data]
        │
        ▼ concept_hgn_model.py: ConceptHGN.forward()
  Module 1: BERT Encoding
  Module 2: CES + 01-GATE
  Module 3: Multi-Level MHSA
  Module 4: R-GAT (4 relations)
  Module 5: Gate Attention → logits [3]
        │
        ▼ train_fever.py: CrossEntropyLoss → backprop
```

## 2. Kiến trúc Concept-HGN (5 modules)

### Module 1 — BERT Encoding

3 loại input per evidence sentence:
```
h_c  = BERT(claim)         [CLS] → [d]   claim representation
h_i  = BERT(claim </s> vi) [CLS] → [N,d] claim-aware evidence
h_vi = BERT(vi)            all tokens → [N, seq_len, d]  cho MHSA
```

### Module 2 — Concept Evidence Selection (CES + 01-GATE)

```
Entity extraction: spaCy NER → entity surface texts
Entity linking:   TAGME API → Wikipedia page title
Concept retrieval: Wikidata P31/P279 → concept label strings
Concept encoding: BERT("<entity> is a <concept>") → [d]

01-GATE:
  sim = softmax(dot(sent_emb, concept_embs.T))   → [num_concepts]
  mask = (sim >= alpha)                            → binary mask
  top-k among survivors
  enriched = sent_emb + sum(filtered_sim * concept_embs)

Hyperparams: alpha=0.8 (FEVER), k=10
```

### Module 3 — Multi-Level Interaction

```
Word-level:
  H̃ = H̄ + MHSA(H̄)   per evidence (residual)
  Extract CLS of each H̃_j → [N, d]

Sentence-level:
  F = stack [CLS_1, ..., CLS_N]        [N, d]
  S = F + MHSA(F)                      [N, d]  ← output S
```

### Module 4 — Heterogeneous Graph + GAT

Node layout per sample:
```
[0]       = claim node          (init: h_c_bar từ CES)
[1..N]    = evidence nodes      (init: h_i_bar từ CES)
[N+1..]   = entity nodes        (init: zeros)
```

4 Edge types:
```
0 = claim–evidence      (bidirectional)
1 = claim–entity        (entities in claim)
2 = evidence–entity     (entities in each evidence)
3 = evidence–evidence   (fully connected)
```

R-GAT propagation (k=2 layers, reuse RGATConv từ HeterFC):
```python
for gnn in self.gnns:
    X = F.relu(gnn(X, edge_index, edge_type))
M = X[1:N+1]  # [N, d]  evidence representations after GAT
```

### Module 5 — Evidence Aggregation + Prediction

```
Gate Attention:
  C    = ReLU(W_c · concat[S, M])     [N, d]
  M̄   = softmax(C) ⊙ M               [N, d]
  gate = Sigmoid(W_g · concat[S, M̄]) [N, d]
  Q    = gate ⊙ Tanh(M̄)             [N, d]
  Q_agg = mean(Q, dim=0)              [d]

Prediction:
  logits = Linear(Q_agg)              [3]
  loss   = CrossEntropyLoss(logits, label)
```

## 3. Bảng tái dụng code HeterFC

| HeterFC | Function | Cách tái dụng |
|---|---|---|
| `train.py` | `setup_seed()` | COPY NGUYÊN |
| `train.py` | `get_optimizer()` | COPY + sửa 1 dòng điều kiện tên layer |
| `train.py` | scheduler, clip_grad, early stopping | COPY NGUYÊN |
| `preprocess.py` | `myset` Dataset class | COPY NGUYÊN |
| `preprocess.py` | pickle cache pattern | COPY NGUYÊN |
| `preprocess.py` | `pack_graph_data()` | COPY + bỏ `gold_indicator` |
| `model.py` | `from torch_geometric.nn import RGATConv` | REUSE import |
| `model.py` | `nn.ModuleList` GNN pattern | TEMPLATE |
| `feverous_scorer.py` | scorer pattern | THAY bằng sklearn accuracy/F1 |

## 4. Cấu trúc thư mục

```
concept_hgn/
├── __init__.py
├── requirements.txt
├── ANALYSIS.md              ← file này
│
├── fever_loader.py          ← đọc FEVER jsonl, normalize evidence
├── concept_extractor.py     ← spaCy + TAGME + Wikidata → concept cache
├── ces_module.py            ← 01-GATE + ConceptEncoder
├── preprocess_fever.py      ← tokenize + sentence-level graph builder
├── concept_hgn_model.py     ← ConceptHGN (5 modules)
└── train_fever.py           ← training loop
```

## 5. Hyperparams từ paper

| Param | Giá trị | Nguồn |
|---|---|---|
| PLM | RoBERTa-Large | paper Section 4 |
| k (top-k concepts) | 10 | paper ablation |
| alpha (01-GATE) | 0.8 (FEVER), 0.7 (UKP) | paper ablation |
| GNN layers | 2 | inferred from architecture |
| LR (PLM) | 1e-5 | paper Section 4 |
| LR (task) | 1e-3 | HeterFC pattern |
| Warmup | 20% | HeterFC |
| Batch size | 4 | HeterFC |
| Epochs | 10 | paper |

## 6. Chuẩn bị dữ liệu

### FEVER dataset
Download pre-processed FEVER với evidence text từ KGAT repo:
```
https://github.com/thunlp/KernelGAT
  → data/fever/train.jsonl
  → data/fever/dev.jsonl
```
Format mỗi dòng: `{"id": int, "claim": str, "label": str, "evidence": [[sent_text, score], ...]}`

### Build concept cache (chạy 1 lần, ~2-4 giờ)
```bash
cd concept_hgn
export TAGME_TOKEN=<your_token>   # https://sobigdata.d4science.org/
python concept_extractor.py --input ../data/fever/dev.jsonl --cache ../data/fever/concept_cache.pkl
python concept_extractor.py --input ../data/fever/train.jsonl --cache ../data/fever/concept_cache.pkl
```

### Cài dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## 7. Chạy training

```bash
cd concept_hgn
python train_fever.py
```

## 8. Điểm rủi ro

| Rủi ro | Xác suất | Mitigation |
|---|---|---|
| TAGME API rate limit | Cao | sleep(0.05) trong extractor + cache trước |
| GPU OOM với RoBERTa-Large + 3 input types | Cao | `gradient_checkpointing=True`, batch_size=2 |
| Wikidata SPARQL timeout | Trung bình | SPARQL_TIMEOUT=10s, retry 1 lần |
| Gate Attention formula mơ hồ (paper thiếu chi tiết) | Thấp | Implementation dựa vào mô tả paper + thực nghiệm |

## 9. Target kết quả (từ paper)

| Metric | Paper | Expected reproduced |
|---|---|---|
| Label Accuracy (FEVER) | 80.26% | ~78–80% |
| FEVER Score | 77.68% | ~75–78% |
| Macro-F1 (UKP Snopes) | 61.9% | N/A (chưa implement) |
