"""
train_fever.py — Training script for Concept-HGN on FEVER.

Reuses from HeterFC train.py (copied verbatim or near-verbatim):
  - setup_seed()
  - get_optimizer()  (1 line changed: layer name check)
  - gradient clipping, accumulation, early stopping pattern
  - scheduler setup

Changes from HeterFC train.py:
  - Loss: CrossEntropyLoss only (no assisted BCELoss — Concept-HGN has no Loss_e)
  - Model: ConceptHGN instead of T2GV2_noEnt
  - Data: FEVER splits instead of FEVEROUS
  - Metrics: Label Accuracy + FEVER Score (3-class)

Hyperparams from paper:
  lr_plm = 1e-5, lr_task = 1e-3, warmup = 20%, epochs = 10
  batch_size = 4, accumulation = 2
"""

import os
import argparse
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from fever_loader import load_fever, LABEL_MAP, IDX_TO_LABEL
from concept_extractor import load_cache, get_sample_concepts
from preprocess_fever import preprocess_fever, myset
from concept_hgn_model import ConceptHGN

import spacy


# ── copied verbatim from HeterFC train.py ───────────────────────────────────
def setup_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model: nn.Module, lr: float, weight_decay: float = 1e-4):
    """
    Differential learning rates:
      PLM params (roberta / encoder layers): lr = lr_plm (1e-5)
      Task params (GNN, CES, MHSA, etc.):   lr = 100 * lr_plm (1e-3)

    Only change from HeterFC: condition checks for "roberta" OR "plm"
    to match ConceptHGN's attribute names.
    """
    plm_params, task_params = [], []
    for name, params in model.named_parameters():
        if 'roberta' in name or 'plm' in name:
            plm_params.append((name, params))
        else:
            task_params.append((name, params))

    print('Task parameters:')
    for name, p in task_params:
        print(f'  {name}: {p.shape}')

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_groups = [
        {
            'params': [p for n, p in plm_params if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay, 'lr': lr,
        },
        {
            'params': [p for n, p in plm_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': lr,
        },
        {
            'params': [p for _, p in task_params],
            'weight_decay': weight_decay, 'lr': 100 * lr,
        },
    ]
    return AdamW(param_groups)
# ────────────────────────────────────────────────────────────────────────────


# ── Config ───────────────────────────────────────────────────────────────────
# All paths can be overridden via environment variables so the script works
# identically on local machines and on vast.ai without editing source code.
#
# On vast.ai, set these in the container env or export before running:
#   export DATA_DIR=/workspace/data
#   export CONCEPT_CACHE=/workspace/concept_cache.pkl
#   export CHECKPOINT_DIR=/workspace/checkpoints

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../HeterFC/concept-hgn
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))    # .../fact-checking-project
_DEFAULT_DATA = os.path.join(_REPO, 'KernelGAT', 'data', 'KernelGAT', 'data')

PLM_PATH      = os.environ.get('PLM_PATH',        'roberta-large')
TRAIN_PATH    = os.environ.get('TRAIN_PATH',      os.path.join(_DEFAULT_DATA, 'all_train.json'))
DEV_PATH      = os.environ.get('DEV_PATH',        os.path.join(_DEFAULT_DATA, 'all_dev.json'))
CONCEPT_CACHE = os.environ.get('CONCEPT_CACHE',   os.path.join(_HERE, 'concept_cache.pkl'))
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', os.path.join(_HERE, 'checkpoint'))
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, 'concept_hgn_fever.pt')

LABEL_MAP_FEVER = {'SUPPORTS': 1, 'REFUTES': 2, 'NOT ENOUGH INFO': 0}

# Hyperparams from paper
LR = 1e-5
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.2
NUM_EPOCHS = 10
TRAIN_BATCH = 1
DEV_BATCH = 1
ACCUM_STEPS = 2
PLM_DIM = 1024              # RoBERTa-Large hidden size
NUM_HEADS = 8
GNN_LAYERS = 2
NUM_RELATIONS = 4
CES_ALPHA = 0.8             # FEVER threshold
CES_TOP_K = 10
EVALUATION_STEPS = 500     # evaluate every 500 steps on GPU
LOGGING_STEPS = 100
# SMOKE_TEST: quick sanity-check on 200/100 samples.
# Override via:  SMOKE_TEST=0 python train_fever.py
#            or: python train_fever.py --smoke
SMOKE_TEST  = bool(int(os.environ.get('SMOKE_TEST', '1')))  # default ON locally
SMOKE_TRAIN = 200
SMOKE_DEV   = 100
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(model, dataloader, concept_data_list, device, split='dev'):
    """Compute Label Accuracy and per-class metrics."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Evaluating {split}'):
            batch = batch.to(device)
            if concept_data_list:
                sidx = int(batch.sample_idx.item())
                cd = [concept_data_list[sidx]]
            else:
                cd = None
            logits = model(batch, cd)
            preds = logits.argmax(dim=-1).cpu().tolist()
            labels = batch.y.cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    la = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'[{split}] Label Accuracy: {la:.4f} | Macro-F1: {macro_f1:.4f}')
    for cls_idx, cls_name in IDX_TO_LABEL.items():
        cls_preds = [p for p, l in zip(all_preds, all_labels) if l == cls_idx]
        cls_labels = [l for l in all_labels if l == cls_idx]
        if cls_labels:
            acc = sum(p == cls_idx for p in cls_preds) / len(cls_labels)
            print(f'  {cls_name}: acc={acc:.4f} ({len(cls_labels)} samples)')

    return la, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Run smoke test on 200/100 samples (overrides SMOKE_TEST env var)')
    parser.add_argument('--no-smoke', action='store_true',
                        help='Disable smoke test — run full training (overrides SMOKE_TEST env var)')
    args = parser.parse_args()

    global SMOKE_TEST
    if args.smoke:
        SMOKE_TEST = True
    elif args.no_smoke:
        SMOKE_TEST = False

    setup_seed(42)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs('preprocessed', exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Load tokenizer + PLM ─────────────────────────────────────────────────
    print('Loading tokenizer and PLM...')
    tokenizer = RobertaTokenizer.from_pretrained(PLM_PATH)
    plm = RobertaModel.from_pretrained(PLM_PATH)

    # ── Load concept cache ────────────────────────────────────────────────────
    concept_cache = {}
    if os.path.exists(CONCEPT_CACHE):
        print(f'Loading concept cache from {CONCEPT_CACHE}')
        concept_cache = load_cache(CONCEPT_CACHE)
    else:
        print(f'Warning: concept cache not found at {CONCEPT_CACHE}. '
              'Run concept_extractor.py first. CES will be disabled.')

    # ── Load spaCy for entity extraction ─────────────────────────────────────
    nlp = None
    if concept_cache:
        print('Loading spaCy model...')
        nlp = spacy.load('en_core_web_lg')

    if not torch.cuda.is_available():
        print('WARNING: No GPU detected. Training on CPU is very slow.')
        print('         Set SMOKE_TEST=True in config for quick iteration.')

    # ── Load + preprocess FEVER data ──────────────────────────────────────────
    print('Loading FEVER train...')
    train_samples = load_fever(TRAIN_PATH)
    print('Loading FEVER dev...')
    dev_samples = load_fever(DEV_PATH)

    if SMOKE_TEST:
        print(f'[SMOKE TEST] Truncating to {SMOKE_TRAIN} train / {SMOKE_DEV} dev samples.')
        train_samples = train_samples[:SMOKE_TRAIN]
        dev_samples = dev_samples[:SMOKE_DEV]

    # Attach entity/concept info to each sample
    def attach_concepts(samples):
        if not concept_cache or nlp is None:
            for s in samples:
                s['claim_entities'] = []
                s['evi_entities'] = [[] for _ in s['evidence_texts']]
            return samples, None

        concept_data_list = []
        for s in tqdm(samples, desc='Attaching concepts'):
            cd = get_sample_concepts(s, concept_cache, nlp)
            s['claim_entities'] = cd['claim_entities']
            s['evi_entities'] = cd['evi_entities']
            concept_data_list.append(cd)
        return samples, concept_data_list

    train_samples, train_concept_data = attach_concepts(train_samples)
    dev_samples, dev_concept_data = attach_concepts(dev_samples)

    print('Preprocessing train...')
    train_data = preprocess_fever(train_samples, tokenizer, split='train')
    print('Preprocessing dev...')
    dev_data = preprocess_fever(dev_samples, tokenizer, split='dev')

    # shuffle=True is safe because we look up concept_data via batch.sample_idx
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=DEV_BATCH, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    model = ConceptHGN(
        plm=plm,
        tokenizer=tokenizer,
        hidden_dim=PLM_DIM,
        num_heads=NUM_HEADS,
        num_relations=NUM_RELATIONS,
        num_class=3,
        gnn_layers=GNN_LAYERS,
        alpha=CES_ALPHA,
        top_k=CES_TOP_K,
    ).to(device)

    optimizer = get_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
    num_training_steps = NUM_EPOCHS * len(train_loader) // ACCUM_STEPS
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(WARMUP_RATIO * num_training_steps),
        num_training_steps=num_training_steps,
    )

    loss_fn = nn.CrossEntropyLoss()  # Concept-HGN: single CE loss (no assisted loss)

    # ── Training ──────────────────────────────────────────────────────────────
    args_table = PrettyTable()
    args_table.field_names = ['Steps', 'Batch', 'Epochs', 'Accum', 'LR', 'α', 'k']
    args_table.add_row([num_training_steps, TRAIN_BATCH, NUM_EPOCHS,
                        ACCUM_STEPS, LR, CES_ALPHA, CES_TOP_K])
    print(args_table)
    print('------ Start Training! ------')

    best_la, best_f1 = 0., 0.
    global_steps = 0
    logging_loss = 0.
    early_stop_cnt = 0
    EARLY_STOP_PATIENCE = 10

    for epoch in range(NUM_EPOCHS):
        model.train()
        for num, batch in enumerate(train_loader):
            batch = batch.to(device)
            # Use sample_idx stored in Data to look up concept_data after DataLoader shuffle
            if train_concept_data:
                sidx = int(batch.sample_idx.item())
                cd = [train_concept_data[sidx]]
            else:
                cd = None

            try:
                logits = model(batch, cd)
                loss = loss_fn(logits, batch.y)
                loss = loss / ACCUM_STEPS
                loss.backward()
                logging_loss += loss.item()
            except RuntimeError:
                torch.cuda.empty_cache()
                try:
                    logits = model(batch, cd)
                    loss = loss_fn(logits, batch.y)
                    loss = loss / ACCUM_STEPS
                    loss.backward()
                    logging_loss += loss.item()
                except Exception as e:
                    print(f'Skipping batch {num}: {e}')
                    optimizer.zero_grad()
                    continue

            if (num + 1) % ACCUM_STEPS == 0:
                # copied verbatim from HeterFC train.py
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=5.0
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_steps += 1

                if global_steps % LOGGING_STEPS == 0:
                    print(f'Epoch {epoch+1} | Step {global_steps} | '
                          f'Loss: {logging_loss / LOGGING_STEPS:.4f}')
                    logging_loss = 0.

                if global_steps % EVALUATION_STEPS == 0:
                    la, f1 = evaluate(model, dev_loader, dev_concept_data, device)
                    model.train()

                    if la > best_la:
                        best_la = la
                        best_f1 = f1
                        early_stop_cnt = 0
                        torch.save(model.state_dict(), MODEL_SAVE_PATH)
                        print(f'  -> Best model saved (LA={best_la:.4f}, F1={best_f1:.4f})')
                    else:
                        early_stop_cnt += 1
                        if early_stop_cnt >= EARLY_STOP_PATIENCE:
                            print(f'Early stopping at step {global_steps}.')
                            break

        else:
            continue
        break

    print(f'\nTraining complete. Best LA={best_la:.4f}, Best Macro-F1={best_f1:.4f}')
    print(f'Model saved to {MODEL_SAVE_PATH}')


if __name__ == '__main__':
    main()
