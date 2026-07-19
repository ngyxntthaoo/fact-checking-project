"""
fever_loader.py — Load FEVER dataset for Concept-HGN fact verification.

FEVER jsonl format (each line):
  {"id": int, "claim": str, "label": str, "evidence": [[[ann_id, evi_id, wiki_page, sent_num], ...]]}

This loader expects pre-processed data with evidence text already extracted,
following the format used by KGAT/GEAR repos (e.g. fever.train.jsonl):
  {"id": int, "claim": str, "label": str, "evidence": [[str, ...], ...]}

Download pre-processed FEVER + evidence text from:
  https://github.com/thunlp/KernelGAT  (kgat_data/fever/)
"""

import json
import jsonlines

LABEL_MAP = {
    'SUPPORTS': 1,
    'REFUTES': 2,
    'NOT ENOUGH INFO': 0,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

MAX_EVIDENCE = 5


def load_fever(path: str, max_evidence: int = MAX_EVIDENCE) -> list[dict]:
    """
    Load FEVER jsonl and return a list of dicts with keys:
      - claim: str
      - label: int  (0=NEI, 1=SUPPORTS, 2=REFUTES)
      - evidence_texts: list[str]  (up to max_evidence sentences)
      - id: int

    Handles two common pre-processed formats:
    Format A (KGAT style): evidence is list of [sent_text, score] pairs
    Format B (raw FEVER + pre-fetched): evidence is list of strings
    """
    samples = []
    with jsonlines.open(path) as reader:
        for line in reader:
            label_str = line.get('label', 'NOT ENOUGH INFO')
            label = LABEL_MAP.get(label_str.upper(), 0)

            raw_evidence = line.get('evidence', [])
            evidence_texts = _extract_evidence_texts(raw_evidence, max_evidence)

            if not evidence_texts:
                evidence_texts = ['No evidence available.']

            samples.append({
                'id': line.get('id', -1),
                'claim': line['claim'],
                'label': label,
                'evidence_texts': evidence_texts,
            })
    return samples


def _extract_evidence_texts(raw_evidence, max_evidence: int) -> list[str]:
    """
    Normalize evidence from multiple possible formats into a flat list of strings.
    """
    texts = []
    seen = set()

    for item in raw_evidence:
        if not item:
            continue

        if isinstance(item, str):
            # Format B: item is already a string
            _add_text(item, texts, seen, max_evidence)

        elif isinstance(item, list):
            if len(item) == 0:
                continue
            first = item[0]

            if isinstance(first, str):
                # Format A: [sent_text, score] or [sent_text]
                _add_text(first, texts, seen, max_evidence)

            elif isinstance(first, list):
                # Raw FEVER nested format: [[ann_id, evi_id, wiki_page, sent_num], ...]
                # Cannot retrieve text without Wikipedia dump — skip
                pass

        if len(texts) >= max_evidence:
            break

    return texts


def _add_text(text: str, texts: list, seen: set, max_evidence: int):
    text = text.strip()
    if text and text not in seen and len(texts) < max_evidence:
        seen.add(text)
        texts.append(text)


def load_fever_splits(train_path: str, dev_path: str, test_path: str = None) -> dict:
    """Convenience wrapper to load all splits at once."""
    splits = {
        'train': load_fever(train_path),
        'dev': load_fever(dev_path),
    }
    if test_path:
        splits['test'] = load_fever(test_path)
    return splits


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/fever/dev.jsonl'
    samples = load_fever(path)
    print(f'Loaded {len(samples)} samples')
    for s in samples[:3]:
        print(f"  [{IDX_TO_LABEL[s['label']]}] {s['claim'][:80]}")
        for e in s['evidence_texts']:
            print(f"    evidence: {e[:80]}")
