import json, gzip, argparse, os
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
from bioc import biocxml
from concurrent.futures import ThreadPoolExecutor

# Increase this based on your OpenAI rate limits (e.g., 5, 10, or 20)
MAX_WORKERS = 10 

def gen_feedback_task(args, client, anno, sentence):
    """Worker function for threading"""
    
    if args.ontology_name:
        ontology = args.ontology_name
    else:
        ontology = anno.infons.get("obo_assignment", "OBO")
    mention = anno.text
    
    prompt = f"""
    You are a biomedical informatics expert. Given the context "{sentence}", specify the standard scientific name from the {ontology} ontology for the concept **{mention}**.
    Format the output using the following JSON structure:
    {{ "name": <"standard scientific concept name"> }}
    """

    try:
        response = client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        if result.get("name"):
            anno.infons['feedback'] = result.get("name")
    except Exception as e:
        print(f"Error for '{mention}': {e}")

def generate_feedback_batch(args, xml_path):
    client = OpenAI(api_key=args.openai_key)
    is_gz = str(xml_path).endswith('.gz')
    opener = gzip.open if is_gz else open
    
    with opener(xml_path, 'rt', encoding='utf-8') as fp:
        collection = biocxml.load(fp)

    # Collect all tasks to be performed
    tasks = []
    for doc in collection.documents:
        for passage in doc.passages:
            for sent in passage.sentences:
                for anno in sent.annotations:
                    tasks.append((anno, sent.text))

    # Execute requests in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(
            tqdm(
                executor.map(lambda p: gen_feedback_task(args, client, p[0], p[1]), tasks), 
                total=len(tasks), desc=f"Annotating {os.path.basename(xml_path)}", leave=False
                )
            )

    # Save output
    write_mode = 'wt' if is_gz else 'w'
    with opener(xml_path, write_mode, encoding='utf-8') as fp:
        biocxml.dump(collection, fp)

def main():
    parser = argparse.ArgumentParser('Annotate BioC XML files in a directory')
    parser.add_argument('--data_dir', required=True, type=str, help='Folder with .xml or .xml.gz files')
    parser.add_argument('--model_name', default="gpt-4.1-mini-2025-04-14", type=str)
    parser.add_argument('--openai_key', required=True, type=str)
    parser.add_argument('--ontology_name', type=str, help='Ontology name to use in the prompt.')

    args = parser.parse_args()

    files = [f for f in os.listdir(args.data_dir) if f.endswith(('.xml', '.gz'))]
    
    for filename in tqdm(files, desc="Processing files"):
        xml_path = os.path.join(args.data_dir, filename)
        try:
            generate_feedback_batch(args, xml_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
