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
KB_DIR=../../kbs
    
python process_ctd_terminology.py \
  --terminology_path  ${KB_DIR}/CTD_disease.tsv.gz \ 
  --output_dir ${KB_DIR}/ctd-disease
```

## 2. Embed the disambiguated terminology 
Use `embed_terms.py` to generate embeddings for all concept aliases in `terminology.json.gz`. \
To build an index from several terminologies, just list the files you want to combine as shown below.

```
KB_DIR=../../kbs
MODEL_NAME=cambridgeltl/SapBERT-from-PubMedBERT-fulltext

python scripts/embed_terms.py \
    --ontology ${KB_DIR}/ctd-diseases/terminology.json.gz 
               ${KB_DIR}/ctd-chemicals/terminology.json.gz 
    --model_name ${MODEL_NAME} \
    --out_vectors ${KB_DIR}/mix_embeddings.npy
```

## 3. Prepare the corpora

For every dataset partition (train/val/test), we split each document into sentences, and re-annotate it at the sentence level. \
Additionally, the annotations will be filtered based on the disambiguated kb (removing oov terms).\
The test set is filtered, removing mentions that exactly overlap with mentions in the train+val sets.

```
DATA_DIR=../../source_corpora
KB_DIR=../../kbs
OUTPUT_DIR=../../processed_data

python prepare_ncbi_disease.py \
    --data_dir ${DATA_DIR}\
    --dictionary_dir ${KB_DIR}/ctd-diseases \
    --output_dir ${OUTPUT_DIR} \
    --with_sentences\
    --filter_test
```

## 4. [OPTIONAL] Run Generative Query Reformulation

The `generate_feedback.py` script generates standard OBO name for each detected mention. It modifies the file inplace, adding entity annotations at the mention level.

```
MODEL_NAME="Qwen/Qwen3-14B"
HF_TOKEN="YOUR_HF_TOKEN"
DATA_DIR=../../processed_data

python generate_feedback_local.py \
    --data_dir ${DATA_DIR} \
    --kb_name "CTD Disease" \
    --model_name ${MODEL_NAME} \
    --hf_token ${HF_TOKEN}     
```

## 5. Retrieve candidates

The following script retrieves 20 candidate concepts from the target terminology for each annotated mention in the xml file.
```
MODEL_NAME=cambridgeltl/SapBERT-from-PubMedBERT-fulltext 
DATA_DIR=../../processed_data
KB_DIR=../../kbs

python retrieve_candidates.py 
--input ${DATA_DIR}/train.xml.gz
--kb_vectors ${KB_DIR}/ctd-disease/embeddings.npy
--model_name ${MODEL_NAME}
--output_file ${DATA_DIR}/train.xml.gz
--top_k 20 --apply_grf --use_rocchio --alpha .6
```
