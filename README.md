# BeLink
Biomedical Entity Linking Meets Generative Re-Ranking

## Installation

```
git clone https://github.com/dash-ka/BeLink.git
cd BeLink
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## 1. Load and disambiguate target terminology

The preprocessing script will generate three files:
- `terminology.json.gz` - A json containing the OBO terminology (e.g. of the structure {"id": "MESH:0001", "name": concept_name})
- `processed_kb.json.gz` - A json with the metadata for obo terms, like id, definition, alt_ids, synonyms
- `statistics_kb.json` - A json file reporting the main statistics for the disambiguated terminology.
- `alt_ids2cui.json` - A json file with alternative ids for each concept.

```bash
cd scripts/preprocess
KB_DIR=../../kbs

python process_ctd_terminology.py \
  --terminology_path ${KB_DIR}/CTD_disease.tsv.gz \
  --output_dir ${KB_DIR}/ctd-disease
```

## 2. Embed the disambiguated terminology

Use `embed_terms.py` to generate embeddings for all concept aliases in `terminology.json.gz`.
To build an index from several terminologies, just list the files you want to combine as shown below.

```bash
KB_DIR=../../kbs
MODEL_NAME=cambridgeltl/SapBERT-from-PubMedBERT-fulltext

python scripts/embed_terms.py \
    --ontology ${KB_DIR}/ctd-diseases/terminology.json.gz \
               ${KB_DIR}/ctd-chemicals/terminology.json.gz \
    --model_name ${MODEL_NAME} \
    --out_vectors ${KB_DIR}/mix_embeddings.npy
```

## 3. Prepare corpora

For every dataset partition (train/val/test), we split each document into sentences and re-annotate it at the sentence level.
Additionally, the annotations will be filtered based on the disambiguated kb (removing oov terms).
The test set is filtered, removing mentions that exactly overlap with mentions in the train+val sets.

```bash
DATA_DIR=../../source_corpora
KB_DIR=../../kbs
OUTPUT_DIR=../../processed_data/ncbi-disease

python prepare_ncbi_disease.py \
    --data_dir ${DATA_DIR} \
    --terminology_dir ${KB_DIR}/ctd-diseases \
    --output_dir ${OUTPUT_DIR} \
    --with_sentences \
    --filter_test
```

## 4. [OPTIONAL] Run Generative Query Reformulation

The `generate_feedback_local.py` script generates a standard scientific name for each detected mention. It modifies the file in place, adding entity annotations at the mention level.

```bash
MODEL_NAME="Qwen/Qwen3-14B"
HF_TOKEN="YOUR_HF_TOKEN"
DATA_DIR=../../processed_data/ncbi-disease

python generate_feedback_local.py \
    --data_dir ${DATA_DIR} \
    --kb_name "CTD Disease" \
    --model_name ${MODEL_NAME} \
    --hf_token ${HF_TOKEN}
```

## 5. Retrieve candidates

The following script retrieves 20 candidate concepts from the target terminology for each annotated mention in the input xml file.

```bash
MODEL_NAME=cambridgeltl/SapBERT-from-PubMedBERT-fulltext
DATA_DIR=../../processed_data/ncbi-disease
KB_DIR=../../kbs

python retrieve_candidates.py \
    --input ${DATA_DIR}/train.xml.gz \
    --kb_vectors ${KB_DIR}/ctd-disease/embeddings.npy \
    --model_name ${MODEL_NAME} \
    --output_file ${DATA_DIR}/train.xml.gz \
    --top_k 20 --apply_grf --use_rocchio --alpha .6
```

## 6. Build HF Dataset for reranker training

```bash
DATA_DIR=../../processed_data/ncbi-disease
KB_DIR=../../kbs
HF_REPO="Name of the dataset repo on HF"
HF_TOKEN="Your HF token"
N_EPOCH="Number of training epochs"

python build_hf_datasets.py \
    --dir_local ${DATA_DIR} \
    --dir_hf_dataset ${HF_REPO} \
    --train_path ${DATA_DIR}/traindev.bioc.xml.gz \
    --test_path ${DATA_DIR}/test.bioc.xml.gz \
    --epoch ${N_EPOCH} \
    --processed_kb_path ${KB_DIR}/processed_kb.json \
    --hf_token ${HF_TOKEN}
```

## 7. Register your dataset with swift

Before running training, you need to register your HuggingFace dataset with the swift framework.
Open `belink/swift_integration.py` and update the `hf_dataset_id` field with the path to your dataset on HuggingFace:

```python
register_dataset(
    DatasetMeta(
        hf_dataset_id='YOUR_HF_USERNAME/YOUR_DATASET_NAME',  # <-- update this
        split=['train'],
        preprocess_func=SelectionPreprocessor(),
        tags=['chat', 'selection'],
    )
)
```

You can verify the dataset was registered successfully by running:

```bash
python -c "
import belink.swift_integration
from swift.llm.dataset.register import DATASET_MAPPING
matches = [k for k in DATASET_MAPPING if any('your_dataset_name' in str(s).lower() for s in k)]
print('Registered:', matches)
"
```

## 8. Train BeLink reranker

```bash
HF_REPO="Name of the dataset repo on HF"
CHECKPOINT_DIR=../../trained_reranker

CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen3-8B \
    --train_type full \
    --num_train_epochs 1 \
    --output_dir ${CHECKPOINT_DIR} \
    --dataset ${HF_REPO} \
    --use_hf 1 \
    --download_mode force_redownload \
    --torch_dtype bfloat16 \
    --dataloader_num_workers 4 \
    --warmup_ratio 0.05 \
    --learning_rate 6e-6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eval_steps 500 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --logging_steps 500
```
