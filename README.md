# BeLink
Biomedical Entity Linking Meets Generative Re-Ranking

## 1. Load and disambiguate a target terminology

The preprocessing script will generate three files:
- `terminology.json.gz` - A json containing the OBO terminology (e.g. of the structure {"id": "MESH:0001", "name": concept_name})
- `processed_kb.json.gz` - A json with the metadata for obo terms, like id, definition, alt_ids, synonyms
- `statistics_kb.json` - A json file reporting the main statistics for the disambiguated terminology.
- `alt_ids2cui.json` - A json file with alternative ids for each concept.
```
cd scripts/preprocess
KB_DIR=./kbs
    
python process_ctd_terminology.py \
  --terminology_path  ${KB_DIR}/CTD_disease.tsv.gz \ 
  --output_dir ${KB_DIR}/ctd-disease
```

## 2. Embed the disambiguated terminology 
Use `embed_terms.py` to generate embeddings for all concept aliases in `terminology.json.gz`.
To build an index from several terminologies, just list the files you want to combine as shown below.

```
KB_DIR=./kbs
MODEL_NAME=cambridgeltl/SapBERT-from-PubMedBERT-fulltext

python scripts/embed_terms.py \
    --ontology ${KB_DIR}/ctd-diseases/terminology.json.gz 
               ${KB_DIR}/ctd-chemicals/terminology.json.gz 
    --model_name ${MODEL_NAME} \
    --out_vectors ${KB_DIR}/mix_embeddings.npy
```

## 3. Prepare the corpora

The script will process each split of the dataset (train/val/test), splitting each document into sentences, and re-annotating it at the sentence level. \
Additionally, the annotations will be filtered based on the disambiguated kb (removing oov terms).\
The test set is filtered, removing mentions that exactly overlap with mentions in the train+val sets.

```
DATA_DIR=./source_corpora
KB_DIR=./kbs
OUTPUT_DIR=./processed_data

python prepare_ncbi_disease.py \
    --data_dir ${DATA_DIR}\
    --dictionary_dir ${KB_DIR}/ctd-diseases \
    --output_dir ${OUTPUT_DIR} \
    --with_sentences\
    --filter_test
```
