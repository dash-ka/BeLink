from tqdm import tqdm
import json, argparse, os, re, gzip
from bioc import biocxml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def extract_json(text):
    output = {"name": ""}
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            # Basic cleanup for common LLM hallucinations
            json_str = re.sub(r':\s*}', '}', json_str.strip())
            if json_str.strip().endswith(":"):
                json_str = json_str.strip() + "}"
            json_dict = json.loads(json_str)
            if isinstance(json_dict, dict):
                return json_dict
        except json.JSONDecodeError:
            print("Error decoding:", json_str)
    return output

def generate_feedback(args, client, tokenizer, sampling_params, xml_path):
    is_gz = str(xml_path).endswith('.gz')
    opener = gzip.open if is_gz else open
    
    with opener(xml_path, 'rt', encoding='utf-8') as f:
        data_collection = biocxml.load(f)

    tasks = [] 

    for doc in data_collection.documents:
        for passage in doc.passages:
            for sentence in passage.sentences:
                for anno in sentence.annotations:
                    ontology = args.target_terminology or anno.infons.get("obo_assignment", "OBO")
                    
                    prompt = f"""You are a biomedical informatics expert. Given the context "{sentence.text}", specify the standard scientific name from the {ontology} ontology for the concept **{anno.text}**.
Format the output using the following JSON structure:
{{ "name": <"standard scientific concept name"> }}"""

                    prompt_token_ids = tokenizer.apply_chat_template(
                    [
                        #{"role": "system", "content": ""},
                        {"role": "user", "content": prompt}
                    ], 
                    enable_thinking=False,
                    add_generation_prompt=True
                    )
                    tasks.append({
                        "anno_obj": anno,
                        "prompt_ids": prompt_token_ids,
                        "success": False
                    })

    if not tasks:
        return

    # --- ATTEMPT 1 ---
    prompts_v1 = [t["prompt_ids"] for t in tasks]
    outputs_v1 = client.generate(prompt_token_ids=prompts_v1, sampling_params=sampling_params)

    failed_indices = []
    for i, output in enumerate(outputs_v1):
        generated_text = output.outputs[0].text
        result = extract_json(generated_text)
        
        if result.get("name") and result["name"].strip():
            tasks[i]["anno_obj"].infons["feedback_local"] = result["name"]
            tasks[i]["success"] = True
        else:
            failed_indices.append(i)

    # --- ATTEMPT 2 (RETRY FOR FAILURES) ---
    if failed_indices:
        print(f"Retrying {len(failed_indices)} failed generations for {os.path.basename(xml_path)}...")
        
        # You can optionally use a slightly different sampling param for retries
        retry_params = sampling_params 
        
        retry_prompts = [tasks[idx]["prompt_ids"] for idx in failed_indices]
        outputs_v2 = client.generate(prompt_token_ids=retry_prompts, sampling_params=retry_params)

        for i, output in enumerate(outputs_v2):
            orig_idx = failed_indices[i]
            generated_text = output.outputs[0].text
            result = extract_json(generated_text)
            
            # Save whatever we got on the second try, even if empty, to avoid infinite loops
            final_name = result.get("name", "").strip()
            tasks[orig_idx]["anno_obj"].infons["feedback_local"] = final_name
            if not final_name:
                print(f"Warning: Permanent failure for entity: {tasks[orig_idx]['anno_obj'].text}")


    # Save output
    write_mode = 'wt' if is_gz else 'w'
    with opener(xml_path, write_mode, encoding='utf-8') as fp:
        biocxml.dump(data_collection, fp)

def main():
    parser = argparse.ArgumentParser(description='Generate standard name for a mention conditioned on the target ontology.')
    parser.add_argument('--data_dir', required=True,type=str,help='Path to the folder with xml files.')
    parser.add_argument('--target_terminology', required=True,type=str,help='The official name of the target ontology.')
    parser.add_argument('--model_name', required=True, type=str ,help='Model name on HuggingFace.')
    parser.add_argument('--hf_token', required=True, type=str, help='Your huggingface token.')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    # vLLM setup - using 19000 max_tokens is dangerous for batching
    client = LLM(model=args.model_name, dtype="float16", trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=150) 
    
    files = [f for f in os.listdir(args.data_dir) if f.endswith(('.xml', '.gz'))]
    
    for filename in tqdm(files, desc="Processing files"):
        xml_path = os.path.join(args.data_dir, filename)
        generate_feedback(args, client, tokenizer, sampling_params, xml_path)

if __name__ == '__main__':
    main()
