import numpy as np
import torch, argparse, gzip, json
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from dense_vectors import make_dense_vectors

def main():
    parser = argparse.ArgumentParser('Apply a transformer model to multiple ontology term names and save as one npy')
    parser.add_argument('--ontology', required=True, type=str, nargs='+', 
                        help='One or more GZipped ontology files (space separated)')
    parser.add_argument('--model_name', required=True, type=str, help='Embedding model to use')
    parser.add_argument('--out_vectors', required=True, type=str, help='A Numpy array with the dense vectors')
    args = parser.parse_args()

    all_onto_texts = []
    all_onto_ids = []
    
    # Iterate through all provided ontology files
    for ont_path in args.ontology:
        print(f"Loading ontology: {ont_path}...")
        try:
             # must be gzipped JSON {"id":"concept_id", "name":"alias_text"}
             with gzip.open(ont_path,'rt') as f:
                ontology = json.load(f)
            
            data = [(e["id"], e['name'].lower()) for e in ontology if e.get('name')]
            
            if data:
                ids, texts = zip(*data)
                print(f"   -> Added {len(texts)} terms from {ont_path}")
                all_onto_texts.extend(texts)
                all_onto_ids.extend(ids)
            else:
                print(f"   -> No valid terms found in {ont_path}")

        except Exception as err:
            print(f"  !! Error loading {ont_path}: {err}")

    print(f"Total terms to vectorize: {len(all_onto_texts)}")

    if not all_onto_texts:
        print("No terms found. Exiting.")
        return

    parent_dir = Path(args.out_vectors).parent
    mapping_file = "ontology_mapping.txt" 
    full_mapping_path = parent_dir / mapping_file
    print(f"Saving ID/Text mapping to {full_mapping_path}...")

    parent_dir.mkdir(parents=True, exist_ok=True)

    with open(full_mapping_path, 'w', encoding='utf-8') as f:
        for oid, otext in zip(all_onto_ids, all_onto_texts):
            f.write(f"{oid}||{otext}\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{device=}")

    print(f"Loading {args.model_name} tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    print("Creating dense vectors of all aggregated ontology names...")
    onto_vectors = make_dense_vectors(model, tokenizer, all_onto_texts)
    print(f"Final shape: {onto_vectors.shape}")

    print(f"Saving to {args.out_vectors}...")
    np.save(args.out_vectors, onto_vectors)

    print("Done.")

if __name__ == '__main__':
    main()
