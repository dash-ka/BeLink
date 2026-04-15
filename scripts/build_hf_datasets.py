from bioc import biocxml
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import login, create_repo, upload_file
from datasets import Dataset, DatasetDict

from collections import defaultdict
import random, re, string, json, gzip, argparse


def prepare_for_selection_test(collection, kb, augment=True):

    data = []
    seen_mentions = set()

    for doc in tqdm(collection.documents):
        for passage in doc.passages:
            for sentence in passage.sentences:
                for anno in sentence.annotations:

                    mention = anno.text.lower()
                    if mention in seen_mentions:
                        continue

                    label = anno.infons["concept_id"]
                    candidates_str = anno.infons.get('candidates', "")
                    if not candidates_str:
                        continue

                    positives, candidate_names, seen_ids = [], [], []

                    candidates = candidates_str.replace("||", ">").split("|")

                    for c_idx in candidates:
                        if ">" not in c_idx:
                            continue

                        # take the 1st retrieved synonym per candidate concept
                        parts = c_idx.split(">", 1)
                        candidate_id, candidate_name = parts[0], parts[1]

                        if candidate_id in seen_ids:
                            continue
                        
                        if augment:
                            # Get preferred name from KB
                            kb_entry = kb.get(candidate_id, {})
                            pref_name = kb_entry.get("name", '').lower()
                            
                            if pref_name:
                                target = f"({pref_name})"
                                if target in candidate_name.lower():
                                    candidate_name =  candidate_name.lower().replace(target, f"(aka {pref_name})")
                                elif (candidate_name.lower().strip() != pref_name):
                                    candidate_name = "{} (aka {})".format(candidate_name, pref_name)

                        # collect all positive synonyms
                        if candidate_id == label:
                            positives.append(candidate_name)

                        if candidate_name not in candidate_names:
                            candidate_names.append(candidate_name)
                        seen_ids.append(candidate_id)

                    # -------- Letters --------
                    # number of in-prompt candidates is cupped to 10
                    candidate_names = candidate_names[:10]
                    num_candidates = len(candidate_names)
                    
                    # Generate letters A, B, C...
                    all_letters = list(string.ascii_uppercase)
                    candidate_letters = all_letters[:num_candidates]
                    none_letter = all_letters[num_candidates]

                    assignment = dict(zip(candidate_letters, candidate_names))
                
                    # -------- Correct letter --------
                    correct_letter = none_letter
                    for letter, name in assignment.items():
                        if name in positives:
                            correct_letter = letter
                            break # Found the match

                    # -------- Options --------
                    options = "\n".join(f"{letter}: {name}" for letter, name in assignment.items())
                    options += f"\n{none_letter}: None of the above."

                    data.append(
                        {
                            "instruction": (
                                f"<Instruct>: Given the context '{sentence.text}', "
                                f"select the correct biomedical concept corresponding "
                                f"to '{anno.text}'. Answer using one of the provided options."
                            ),
                            "input": f"<Options>: {options}",
                            "response": correct_letter
                        }
                    )

                    seen_mentions.add(mention)

    print(f"Total samples created: {len(data)}")
    return data

def prepare_for_selection_train(collection, kb, augment=True, epoch=3):

    correct_tracker = defaultdict(int)
    len_tracker = defaultdict(int)
    data = []

    all_letters = list(string.ascii_uppercase)

    for _ in range(epoch):

        for doc in tqdm(collection.documents):
            for passage in doc.passages:
                for sentence in passage.sentences:
                    for anno in sentence.annotations:

                        label = anno.infons["concept_id"]
                        candidates_str = anno.infons.get('candidates', "")
                        candidates = candidates_str.replace("||", ">").split("|")
                    
                        negatives_by_id = defaultdict(list)
                        seen_names, positives = [], []

                        for c_idx in candidates:
                            if ">" not in c_idx: continue
                            parts = c_idx.split(">", 1)
                            candidate_id, candidate_name = parts[0], parts[1]

                            if augment:
                            # Get preferred name from KB
                                kb_entry = kb.get(candidate_id, {})
                                pref_name = kb_entry.get("name", '').lower()
                                
                                if pref_name:
                                    target = f"({pref_name})"
                                    if target in candidate_name.lower():
                                        candidate_name = re.sub(re.escape(target), f"(aka {pref_name})", candidate_name, flags=re.I)
                                    elif (candidate_name.lower().strip() != pref_name):
                                        candidate_name = "{} (aka {})".format(candidate_name, pref_name)
                            
                            # collect all positive synonyms
                            if candidate_id == label:
                                positives.append(candidate_name)
                            elif candidate_name not in seen_names:
                                negatives_by_id[candidate_id].append(candidate_name)
                            seen_names.append(candidate_name)

                        # -------- Build candidate list --------

                        is_nota_trap = random.random() < 0.10 

                        final_candidates = []
                        selected_positive = False

                        if positives and not is_nota_trap:
                            # NORMAL CASE: Positive is included
                            selected_positive = random.choice(positives)
                            final_candidates.append(selected_positive)

                            # Sample up to 9 negatives (or fewer if not enough)
                            num_neg = min(9, len(negatives_by_id))
                            for key in random.sample(list(negatives_by_id), k=num_neg):
                                final_candidates.append(random.choice(negatives_by_id[key]))
                            
                        else:
                            # NOTA CASE: Either there were no positives, or we are hiding the positive
                            # Sample up to 10 negatives only
                            num_neg = min(10, len(negatives_by_id))
                            for key in random.sample(list(negatives_by_id), k=num_neg):
                                final_candidates.append(random.choice(negatives_by_id[key]))
                        
                        # DEDUPLICATE while preserving the positive
                        final_candidates = list(dict.fromkeys(final_candidates))
                        random.shuffle(final_candidates)

                        len_tracker[len(final_candidates)]+=1

                        # -------- Letters --------
                        num_candidates = len(final_candidates)

                        # Generate letters A, B, C...
                        assignment = {all_letters[i]: name for i, name in enumerate(final_candidates)}
                        none_letter = all_letters[num_candidates]
                    
                        # -------- Correct letter --------
                        if selected_positive and selected_positive in final_candidates:
                            correct_letter = [letter for letter, name in assignment.items() if name == selected_positive][0]
                        else:
                            correct_letter = none_letter

                        correct_tracker[correct_letter] +=1

                        # -------- Options --------
                        options = "\n".join(f"{letter}: {name}" for letter, name in assignment.items())
                        options += f"\n{none_letter}: None of the above."

                        data.append(
                            {
                                "instruction": (
                                    f"<Instruct>: Given the context '{sentence.text}', "
                                    f"select the correct biomedical concept corresponding "
                                    f"to '{anno.text}'. Answer using one of the provided options."
                                ),
                                "input": f"<Options>: {options}",
                                "response": correct_letter
                            }
                        )


    print(f"Dataset Size: {len(data)}")
    print(sorted(correct_tracker.items(), key=lambda x:x[-1]))
    return data     

def save_jsonl_gz(dataset, path):
    """
    Serializes a HuggingFace Dataset into a compressed JSONL file.
    
    Args:
        dataset (Dataset): The HF Dataset to save.
        path (Path): Path to the output .jsonl.gz file.
    """
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def build_hf_dataset(args):
    """
    Main pipeline: Loads BioC data, prepares MC-style training/test sets, 
    and uploads them to the Hugging Face Hub.
    """
    # Authenticate with HF
    if args.hf_token:
        login(token=args.hf_token)

    output_dir = Path(args.dir_local)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Repository
    create_repo(repo_id=args.dir_hf_dataset, repo_type="dataset", exist_ok=True)

    # Load Knowledge Base 
    print("Loading Knowledge Base...")
    with open(args.processed_kb_path, 'r', encoding="utf-8") as f:
        kb = json.load(f)

    # Load and Prepare Train Data
    print("Preparing Training Data...")
    is_gz = str(args.train_path).endswith('.gz')
    opener = gzip.open if is_gz else open
    
    with opener(args.train_path, 'rt', encoding='utf-8') as fp:
        train_collection = biocxml.load(fp)
    traindev_data = prepare_for_selection_train(train_collection, kb, epoch=args.epoch)

    # Load and Prepare Test Data
    print("Preparing Test Data...")
    
    is_gz = str(args.test_path).endswith('.gz')
    opener = gzip.open if is_gz else open
    
    with opener(args.test_path, 'rt', encoding='utf-8') as fp:
        test_collection = biocxml.load(fp)
    test_data = prepare_for_selection_test(test_collection, kb)

    # Create HuggingFace DatasetDict
    ds = DatasetDict({
        "train": Dataset.from_list(traindev_data),
        "test": Dataset.from_list(test_data),
    })

    # Save local compressed files
    train_local_path = output_dir / "train.jsonl.gz"
    test_local_path = output_dir / "test.jsonl.gz"
    
    save_jsonl_gz(ds["train"], train_local_path)
    save_jsonl_gz(ds["test"], test_local_path)

    # Method 1: Upload raw JSONL.GZ files for manual download
    for filename, local_path in [("train.jsonl.gz", train_local_path), 
                                 ("test.jsonl.gz", test_local_path)]:
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=args.dir_hf_dataset,
            repo_type="dataset",
        )

    # Push to Hub as an accessible HF Dataset (Parquet format)
    # This makes the dataset viewable/loadable via load_dataset()
    ds.push_to_hub(args.dir_hf_dataset)
    print(f"Successfully pushed dataset to: https://huggingface.co/datasets/{args.dir_hf_dataset}")

def main():
    parser = argparse.ArgumentParser(description='Build and upload a Multiple Choice Biomedical Dataset.')
    parser.add_argument('--dir_local', required=True, type=str, help='Local directory to store temporary files.')
    parser.add_argument('--dir_hf_dataset', required=True, type=str, help='HF Repo ID (e.g., "username/my-dataset").')
    parser.add_argument('--train_path', required=True, type=str, help='Path to training BioC XML.')
    parser.add_argument('--test_path', required=True, type=str, help='Path to testing BioC XML.')
    parser.add_argument('--epoch', default=5, type=int, help='Number of training epochs (shuffled versions).')
    parser.add_argument('--processed_kb_path', required=True, type=str, help='Path to the knowledge base JSON.')
    parser.add_argument('--hf_token', type=str, default=None, help='HF Access Token.')
    
    args = parser.parse_args()

    build_hf_dataset(args)

if __name__ == "__main__":
    main()