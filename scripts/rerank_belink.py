import os, gzip, string, argparse
from bioc import biocxml
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, get_template
from tqdm import tqdm
import belink.swift_integration 

# generation_config
MAX_NEW_TOKENS = 20
TEMPERATURE = 0

def rerank_candidates(engine, xml_path):
    is_gz = str(xml_path).endswith('.gz')
    opener = gzip.open if is_gz else open
    
    with opener(xml_path, 'rt', encoding='utf-8') as fp:
        collection = biocxml.load(fp)

    # 1. Collect all annotation tasks and build requests
    tasks = [] # To keep track of which annotation object to update
    infer_requests = []
    lookup_tables = [] # Stores the mapping for each specific request

    for doc in tqdm(collection.documents, desc="Reranking deocuments"):
        for passage in doc.passages:
            for sent in passage.sentences:
                for anno in sent.annotations:
                    candidates_str = anno.infons.get('candidates', "")
                    if not candidates_str:
                        continue
                        
                    candidates = candidates_str.replace("||", ">").split("|")[:10]
                    num_candidates = len(candidates)
                    letters = list(string.ascii_uppercase[:num_candidates + 1])
                    candidate_letters = letters[:-1]
                    none_letter = letters[-1]
                    
                    # Create a lookup: {'A': {'id': '123', 'name': 'Flu'}, ...}
                    seen_ids = set()
                    current_lookup = {}
                    # keep only the first retrieved alias per candidate concept
                    for c_str in candidates:
                        if ">" not in c_str: continue
                        c_id, c_name = c_str.split(">", 1)
                        if c_id in seen_ids: continue

                        current_lookup[candidate_letters[len(seen_ids)]] = {"id": c_id, "name": c_name}
                        seen_ids.add(c_id)

                    # 2. Build the options string for the prompt
                    options = "\n".join(f"{l}: {data['name']}" for l, data in current_lookup.items())
                    options += f"\n{none_letter}: None of the above."

                    user_content = (
                        f"<Instruct>: Given the context '{sent.text}', "
                        f"select the correct biomedical concept corresponding "
                        f"to '{anno.text}'. Answer using one of the provided options.\n"
                        f"<Options>: {options}"
                    )
                    
                    tasks.append(anno)
                    lookup_tables.append((current_lookup, none_letter))
                    infer_requests.append(InferRequest(messages=[{"role": "user", "content": user_content}]))

    # 2. Batch Inference 
    if infer_requests:
        request_config = RequestConfig(max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
        # SWIFT's engine.infer handles batching internally if a list is passed
        resp_list = engine.infer(infer_requests, request_config)

        # 3. Map responses back to annotations
        for anno, resp, (lookup, none_char) in zip(tasks, resp_list, lookup_tables):
            response_text = resp.choices[0].message.content
            # Improved extraction logic based on your response_prefix
            pred_option = response_text.lower().split("answer")[-1].replace(":", "").strip()
            
            # Use the first character in case the model writes "A: Influenza"
            pred_letter = pred_option[0].upper() if pred_option else ""
            if pred_letter == none_char:
                anno.infons['reranker'] = "None of the above"
            elif pred_letter in lookup:
                anno.infons['reranker'] = "||".join([lookup[pred_letter]['id'], lookup[pred_letter]['name']])
            else:
                # Fallback if the LLM hallucinates a letter
                print("Hallucination ? >>> ", pred_option)
                anno.infons['reranker'] = pred_option

    # 4. Save/Overwrite the XML
    write_mode = 'wt' if is_gz else 'w'
    with opener(xml_path, write_mode, encoding='utf-8') as fp:
        biocxml.dump(collection, fp)
    
    print(f"Finished processing {len(infer_requests)} annotations in {xml_path}")

def main():
    parser = argparse.ArgumentParser('Annotate BioC XML files')
    parser.add_argument('--input_path', required=True, type=str)
    parser.add_argument('--model_name', required=True, type=str)
    args = parser.parse_args()

    # Initialize Engine
    # Note: Ensure "qwen3" is the correct template for your specific model version
    engine = PtEngine(args.model_name, use_hf=True) 
    
    template = get_template(
        engine.model_meta.template, 
        engine.processor, 
        default_system=None, 
        response_prefix="<think>\n\n</think>\n\nAnswer"
    )
    engine.default_template = template
    
    rerank_candidates(engine, args.input_path)

if __name__ == "__main__":
    main()
