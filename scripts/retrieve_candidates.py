
import numpy as np
import gzip, json, torch, argparse
from bioc import biocxml
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from dense_vectors import make_dense_lookup

def get_annotation_texts(collection, obo_prefix):
    """Collects all unique query texts and their associated metadata."""
    annotation_texts = {}
    for doc in collection.documents:
        for passage in doc.passages:
            for s in passage.sentences:
                for anno in s.annotations:
                    clean_mention = anno.text.lower().strip()
                    # --- debug
                    #if not anno.infons.get("feedback", ""):
                    #    print("No feedback available for mention ", anno.text)
                    # ----
                    if clean_mention not in annotation_texts:
                        if obo_prefix and obo_prefix not in anno.infons.get("obo_assignment", "").lower():
                            continue
                
                        annotation_texts[clean_mention] = {
                                "mention": clean_mention, 
                                "feedback": anno.infons.get("feedback", "")
                            }
    return annotation_texts


def main():
    parser = argparse.ArgumentParser('Retrieve candidates and save them directly into BioC XML')
    parser.add_argument('--input', required=True, type=str, help='Gzipped BioC XML data with queries')
    parser.add_argument('--kb_vectors', required=True, type=str, help='Precalculated vectors')
    parser.add_argument('--top_k', required=False, type=int, default=10, help='Max candidates')
    parser.add_argument('--model_name', required=True, type=str, help='Transformer model')
    parser.add_argument('--output_file', required=True, type=str, help='Output Path for the modified BioC XML (gzipped)')
    parser.add_argument('--apply_grf', action="store_true")
    parser.add_argument('--use_rocchio', action="store_true")
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--obo_prefix', type=str) #defaults to None
    args = parser.parse_args()

    # 1. Load BioC Collection
    print(f"Loading BioC XML: {args.input}")
    is_gz = str(args.input).endswith('.gz')
    opener = gzip.open if is_gz else open
    mode = 'rt' if is_gz else 'r'

    with opener(args.input, mode, encoding='utf-8') as fp:
        collection = biocxml.load(fp)

    # 2. Get unique mentions for efficient vectorization
    anno_texts_dict = get_annotation_texts(collection, args.obo_prefix)
    print(f"Loading {len(anno_texts_dict)} unique queries for prefix {args.obo_prefix}.")

    # 3. Setup Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    
    # 4. Load Vectors & Search
    print("Loading ontology vectors and performing search...")
    onto_vectors = np.load(args.ontology_vectors)
    
    lookup = make_dense_lookup(
        model, tokenizer, onto_vectors, anno_texts_dict, 
        args.top_k, with_grf=args.apply_grf, 
        use_rocchio=args.use_rocchio, alpha=args.alpha
    )
    print(f"Retrieved candidates for {len(lookup)} queries.")

    # 4.5 load ontology id to text mapping
    mapping_path = Path(args.ontology_vectors).parent / "ontology_mapping.txt"

    with open(mapping_path, "r", encoding="utf-8") as f:
        ontology_entries = [l.strip() for l in f if l.strip()]
    
    assert len(ontology_entries)==len(onto_vectors)

    # 5. Inject candidates back into the BioC Objects
    print("Injecting candidates into BioC annotations...")
    
    key = "candidates_"+ args.obo_prefix if args.obo_prefix else "candidates"
    for doc in collection.documents:
        for passage in doc.passages:
            for s in passage.sentences:
                for anno in s.annotations:
                    mention_key = anno.text.lower().strip()
                    if mention_key in lookup:
                        # Convert idx to int() so it can index the ontology_names list
                        indices = [int(idx) for idx in lookup[mention_key].split("|")]
                        candidate_names = "|".join([ontology_entries[idx] for idx in indices])
                        anno.infons[key] = candidate_names

    # 6. Save the enriched XML
    print(f"Saving enriched BioC XML to {args.output_file}...")
    write_mode = 'wt' if is_gz else 'w'
    with opener(args.output_file, write_mode, encoding='utf-8') as fp:
        biocxml.dump(collection, fp)

if __name__ == "__main__":
    main()
