"""
concept_extractor.py — Extract entity concepts for Concept-HGN.

Pipeline (run once offline, results cached to .pkl):
  1. Entity extraction: spaCy NER (en_core_web_lg)
  2. Entity linking: TAGME API → Wikipedia page title
  3. Concept retrieval: Wikidata P31 (instance of) + P279 (subclass of)
     — This substitutes for YAGO used in the original paper.
     — Same semantic purpose: get the type/class hierarchy of an entity.
  4. Cache: {entity_surface_text: ["concept label 1", "concept label 2", ...]}

Usage:
  python concept_extractor.py --input data/fever/dev.jsonl --cache data/fever/concept_cache.pkl
"""

import argparse
import pickle
import time
import logging
from pathlib import Path

import spacy
import tagme
from SPARQLWrapper import SPARQLWrapper, JSON

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# TAGME: set your GCUBE token from https://sobigdata.d4science.org/
# export TAGME_TOKEN=<your_token>  or set below
TAGME_TOKEN = None  # override via env var TAGME_TOKEN or set directly

WIKIDATA_ENDPOINT = 'https://query.wikidata.org/sparql'
WIKIDATA_USER_AGENT = 'ConceptHGN-research/1.0 (contact: your@email.com)'

TAGME_THRESHOLD = 0.1   # entity linking confidence threshold
MAX_CONCEPTS_PER_ENTITY = 20
SPARQL_TIMEOUT = 10


def _init_tagme():
    import os
    token = TAGME_TOKEN or os.environ.get('TAGME_TOKEN', '')
    if not token:
        log.warning('TAGME_TOKEN not set. Entity linking will be skipped.')
        return False
    tagme.GCUBE_TOKEN = token
    return True


def extract_entities(text: str, nlp) -> list[str]:
    """Use spaCy NER to extract named entity surface texts from a sentence."""
    doc = nlp(text)
    entities = []
    seen = set()
    for ent in doc.ents:
        surface = ent.text.strip()
        if surface and surface not in seen:
            seen.add(surface)
            entities.append(surface)
    return entities


def link_entity_to_wikipedia(surface: str) -> str | None:
    """Link a surface entity to a Wikipedia page title using TAGME."""
    try:
        annotations = tagme.annotate(surface)
        if annotations is None:
            return None
        best = None
        best_score = TAGME_THRESHOLD
        for ann in annotations.get_annotations(TAGME_THRESHOLD):
            if ann.score > best_score:
                best_score = ann.score
                best = ann.entity_title
        return best
    except Exception as e:
        log.debug(f'TAGME error for "{surface}": {e}')
        return None


def get_wikidata_concepts(wiki_title: str) -> list[str]:
    """
    Query Wikidata for P31 (instance of) and P279 (subclass of) labels
    for a Wikipedia page title. Returns a list of concept label strings.
    """
    sparql = SPARQLWrapper(WIKIDATA_ENDPOINT, agent=WIKIDATA_USER_AGENT)
    sparql.setTimeout(SPARQL_TIMEOUT)

    query = f"""
    SELECT DISTINCT ?conceptLabel WHERE {{
      ?entity wikibase:sitelinks ?sitelinks .
      ?entity schema:name "{wiki_title.replace('"', '')}"@en .
      {{
        ?entity wdt:P31 ?concept .
      }} UNION {{
        ?entity wdt:P279 ?concept .
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
      BIND(COALESCE(?conceptLabel, "") AS ?conceptLabel)
    }}
    LIMIT {MAX_CONCEPTS_PER_ENTITY}
    """

    # Simpler fallback query using sitelinks
    query_v2 = f"""
    SELECT DISTINCT ?conceptLabel WHERE {{
      ?entity ^schema:about ?article .
      ?article schema:inLanguage "en" ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name "{wiki_title.replace('"', '')}@en" .
      {{
        ?entity wdt:P31 ?concept .
      }} UNION {{
        ?entity wdt:P279 ?concept .
      }}
      ?concept rdfs:label ?conceptLabel .
      FILTER(LANG(?conceptLabel) = "en")
    }}
    LIMIT {MAX_CONCEPTS_PER_ENTITY}
    """

    # Use MediaWiki title lookup via Wikidata API as primary approach
    return _query_wikidata_by_title(wiki_title)


def _query_wikidata_by_title(wiki_title: str) -> list[str]:
    """
    Lookup Wikidata entity by Wikipedia title, then get P31+P279 labels.
    Uses Wikidata SPARQL with schema:about pattern.
    """
    sparql = SPARQLWrapper(WIKIDATA_ENDPOINT, agent=WIKIDATA_USER_AGENT)
    sparql.setTimeout(SPARQL_TIMEOUT)

    encoded_title = wiki_title.replace('"', '\\"').replace('\\', '\\\\')
    query = f"""
    SELECT DISTINCT ?conceptLabel WHERE {{
      ?entity ^schema:about ?article .
      ?article schema:inLanguage "en" ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name "{encoded_title}" .
      {{
        ?entity wdt:P31 ?concept .
      }} UNION {{
        ?entity wdt:P279 ?concept .
      }}
      ?concept rdfs:label ?conceptLabel .
      FILTER(LANG(?conceptLabel) = "en")
    }}
    LIMIT {MAX_CONCEPTS_PER_ENTITY}
    """

    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        concepts = []
        for r in results['results']['bindings']:
            label = r.get('conceptLabel', {}).get('value', '').strip()
            if label:
                concepts.append(label)
        return concepts
    except Exception as e:
        log.debug(f'Wikidata SPARQL error for "{wiki_title}": {e}')
        return []


def get_concepts_for_entity(surface: str, tagme_available: bool, nlp=None) -> list[str]:
    """Full pipeline: surface text → Wikipedia title → Wikidata concepts."""
    if tagme_available:
        wiki_title = link_entity_to_wikipedia(surface)
        time.sleep(0.05)  # rate limiting
    else:
        # Fallback: treat surface as Wikipedia title directly
        wiki_title = surface

    if not wiki_title:
        return []

    concepts = _query_wikidata_by_title(wiki_title)
    time.sleep(0.1)  # rate limiting for Wikidata
    return concepts


def build_concept_cache(samples: list[dict], cache_path: str, nlp=None) -> dict:
    """
    Build a concept cache dict: {entity_surface: [concept_label, ...]}
    Processes all claims and evidences in the dataset.

    Args:
        samples: output of fever_loader.load_fever()
        cache_path: where to save the .pkl cache
        nlp: spaCy model (loaded if None)

    Returns:
        cache dict
    """
    if nlp is None:
        log.info('Loading spaCy model...')
        nlp = spacy.load('en_core_web_lg')

    tagme_available = _init_tagme()
    cache = {}
    all_texts = []

    for sample in samples:
        all_texts.append(sample['claim'])
        all_texts.extend(sample['evidence_texts'])

    log.info(f'Extracting entities from {len(all_texts)} texts...')
    all_entities = set()
    for text in all_texts:
        for ent in extract_entities(text, nlp):
            all_entities.add(ent)

    log.info(f'Found {len(all_entities)} unique entities. Fetching concepts...')
    for i, entity in enumerate(all_entities):
        if entity in cache:
            continue
        concepts = get_concepts_for_entity(entity, tagme_available, nlp)
        cache[entity] = concepts
        if (i + 1) % 100 == 0:
            log.info(f'  {i+1}/{len(all_entities)} entities processed')
            _save_cache(cache, cache_path)

    _save_cache(cache, cache_path)
    log.info(f'Concept cache saved to {cache_path} ({len(cache)} entities)')
    return cache


def _save_cache(cache: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(cache, f)


def load_cache(cache_path: str) -> dict:
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def get_sample_concepts(sample: dict, cache: dict, nlp) -> dict:
    """
    For a single sample, return:
      {
        'claim_entities': [str],
        'claim_concepts': {entity: [concept_str]},
        'evi_entities': [[str] per evidence],
        'evi_concepts': [{entity: [concept_str]} per evidence],
      }
    """
    claim_entities = extract_entities(sample['claim'], nlp)
    claim_concepts = {e: cache.get(e, []) for e in claim_entities}

    evi_entities = []
    evi_concepts = []
    for evi_text in sample['evidence_texts']:
        ents = extract_entities(evi_text, nlp)
        evi_entities.append(ents)
        evi_concepts.append({e: cache.get(e, []) for e in ents})

    return {
        'claim_entities': claim_entities,
        'claim_concepts': claim_concepts,
        'evi_entities': evi_entities,
        'evi_concepts': evi_concepts,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _DATA_ROOT = '/Users/thnhthao/Master 2025/Thesis/FC-project/KernelGAT/data/KernelGAT/data'
    _HERE = '/Users/thnhthao/Master 2025/Thesis/FC-project/HeterFC/concept-hgn'
    parser.add_argument(
        '--input',
        default=f'{_DATA_ROOT}/bert_dev.json',
        help='KernelGAT bert_train.json or bert_dev.json path',
    )
    parser.add_argument('--cache', default=f'{_HERE}/concept_cache.pkl')
    args = parser.parse_args()

    import sys
    sys.path.insert(0, '.')
    from fever_loader import load_fever
    samples = load_fever(args.input)
    log.info(f'Loaded {len(samples)} samples from {args.input}')
    build_concept_cache(samples, args.cache)
