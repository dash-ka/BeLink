"""
BeLink custom dataset registration for ms-swift.

This module registers the BeLink selection dataset with swift's dataset registry.
Import this module before any swift training or inference to ensure the dataset
is available.

Usage (already done in scripts/rerank_belink.py):
    import belink.swift_integration  # noqa: F401
"""

from typing import Any, Dict

from swift.llm.dataset.preprocessor import ResponsePreprocessor
from swift.llm.dataset.register import DatasetMeta, register_dataset


class SelectionPreprocessor(ResponsePreprocessor):
    """
    Preprocessor for the BeLink candidate selection dataset.

    Formats each row into a query/response pair for reranking training:
    - query   = instruction + newline + input + optional suffix
    - response = the original response field
    """

    def __init__(self, *args, query_suffix: str = '', response_prefix: str = '', **kwargs):
        self.query_suffix = query_suffix
        self.response_prefix = response_prefix
        super().__init__(*args, **kwargs)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row['query'] = row['instruction'] + "\n" + row['input'] + self.query_suffix
        row['response'] = row['response']
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        hf_dataset_id='Dash00/bc5cdr-disease-sapbert-selection',
        split=['train'],
        preprocess_func=SelectionPreprocessor(),
        tags=['chat', 'selection'],
    )
)

register_dataset(
    DatasetMeta(
        hf_dataset_id='Dash00/ncbi-disease-sapbert-selection',
        split=['train'],
        preprocess_func=SelectionPreprocessor(),
        tags=['chat', 'selection']
    )
)

register_dataset(
    DatasetMeta(
    hf_dataset_id='Dash00/nlm-gene-sapbert-selection',
    split=['train'],
    preprocess_func=SelectionPreprocessor(),
    tags=['chat', 'selection']))

register_dataset(
    DatasetMeta(
    hf_dataset_id='Dash00/gnormplus-sapbert-selection',
    split=['train'],
    preprocess_func=SelectionPreprocessor(),
    tags=['chat', 'selection']))

register_dataset(
    DatasetMeta(
    hf_dataset_id='Dash00/linnaeus-sapbert-selection',
    split=['train'],
    preprocess_func=SelectionPreprocessor(),
    tags=['chat', 'selection']))

register_dataset(
    DatasetMeta(
    hf_dataset_id='Dash00/s800-sapbert-selection',
    split=['train'],
    preprocess_func=SelectionPreprocessor(),
    tags=['chat', 'selection']))

register_dataset(
    DatasetMeta(
    hf_dataset_id='Dash00/bc5cdr-chemical-sapbert-selection',
    split=['train'],
    preprocess_func=SelectionPreprocessor(),
    tags=['chat', 'selection']))

register_dataset(
    DatasetMeta(
    hf_dataset_id='Dash00/nlm-chem-sapbert-selection',
    split=['train'],
    preprocess_func=SelectionPreprocessor(),
    tags=['chat', 'selection']))
