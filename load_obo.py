#!/usr/bin/env python

import argparse, pronto, re, json, requests, gzip
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


def load_obo(path_to_obo_file):
    """
    Loads an OBO ontology. If the file doesn't exist locally, 
    attempts to download it from PURL based on the filename.
    """

    def repair_obo_content(file_path):
        """
        Removes problematic escape characters in xrefs that cause Pronto to crash.
        """
        print(f"Repairing syntax in {file_path.name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 1. Create a "Clean" version of the file without any xref lines
        with open(file_path, 'w', encoding='utf-8') as fout:
            for line in lines:
                # If the line starts with 'xref:', we simply don't write it to the new file
                if not line.startswith("xref:"):
                    fout.write(line)

    path = Path(path_to_obo_file)
    
    if not path.exists():
        obo_prefix = path.stem 
        url = f"http://purl.obolibrary.org/obo/{obo_prefix}.obo"
    
        print(f"File not found locally. Fetching from URL: {url}")
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")

    repair_obo_content(path)

    return pronto.Ontology(str(path))


def build_kb(path_to_obo_file) -> dict:
    """Parse the content into a structured KB dictionary."""

    print("Loading the obo ontology...")
    onto = load_obo(path_to_obo_file)
    print(f"Loaded: {onto.metadata.ontology}")

    # Extract prefix (e.g., 'NCBITaxon') for ID filtering
    OBO_PREFIX = Path(path_to_obo_file).stem.lower() 
    
    kb = {}
    for term in tqdm(onto.terms(), desc="Building KB"):
        # Check if the ID starts with our prefix and is not obsolete
        if OBO_PREFIX in term.id.lower() and not term.obsolete:
            if not term.name:
                continue

            cui = term.id.strip()
            # Filter synonyms: remove InChI keys and junk characters
            syns = {
                s.description.strip() for s in term.synonyms 
                if all(x not in s.description.lower() for x in ["inchikey", "inchi"]) 
                and s.description not in ["[*-]", "."]
            }

            kb[cui] = {
                "cui": cui, 
                "alt_cui": "|".join([t.strip() for t in term.alternate_ids]),
                "name": term.name.strip(), 
                "synonyms": "|".join(list(syns)), 
            } 

    print(f"Built KB with {len(kb)} entries.")
    return kb

def get_name2cuis_mapping(kb: dict) -> tuple:

    alt_ids2cui, new_kb = dict(), dict()
    name2cui = defaultdict(list)
    pattern = re.compile("[+;]")

    for cui, entry in tqdm(kb.items(), total=len(kb), desc="Processing KB"):
        if cui == "1":
            continue

        alt_cuis = entry["alt_cui"].split("|")

        # Normalize CUI by removing composite identifiers
        pattern = re.compile("[+;,]")
        cui = pattern.sub("|", cui)
        if "|" in cui:
            composite = [c.strip() for c in cui.split("|") if c.strip()]
            cui, extras = composite[0], composite[1:]
            alt_cuis.extend(extras)
            print("Composite CUI resolved:", composite)

        # Map alternative IDs
        for alt in alt_cuis:
            if alt != cui:
                alt_ids2cui[alt] = cui

        pref_name = entry["name"].lower()
        if not pref_name:
            continue

        name2cui[pref_name].append(cui)
        synonyms = []
        for syn in entry["synonyms"].split("|"):
            syn = syn.lower().strip()
            if not syn or syn == pref_name:
                continue
            if len(syn) < 5:
                syn = f"{syn} ({pref_name})"
            
            name2cui[syn].append(cui)
            synonyms.append(syn)

        new_kb[cui] = {"name":pref_name, "synonyms":synonyms}

    print(f"Old KB #entities: {len(kb)}\nNew KB # entities: {len(new_kb)}" )
    return name2cui, new_kb, alt_ids2cui

def resolve_homonyms(name2cui: list, kb: dict, alt_ids2cui: dict) -> dict:
    
    """Resolve duplicate term names mapping to different CUIs."""
    
    homonyms, failed = 0, 0
    avg_names = []
    is_homonym = lambda x: len(set(name2cui.get(x, []))) > 1
    disambiguated_aliases = dict()

    for cui, entity in tqdm(kb.items(), desc="Resolving homonyms"):
        
        cui = alt_ids2cui.get(cui, cui)
        pref_name = entity["name"]
        synonyms = entity.get("synonyms")
        synonyms = sorted(set(synonyms) if synonyms else [], key=len, reverse=True)
        
        avg_names.append(len([pref_name] + synonyms))
        for name in [pref_name] + synonyms:
            # find homonyms (same name, different CUI)
            if is_homonym(name): 
                
                homonyms += 1
            # If the preferred name is different, disambiguate
            if name != pref_name:
                new_name = f"{name} ({pref_name})"

                # If the "name + preferred name" exists, disambiguate using synonyms
                if is_homonym(new_name):
                    for s in synonyms:
                            if s not in [name, pref_name]:
                                new_name = f"{name} ({s})"
                                break
                name = new_name
            
            # If preferred name is the same, disambiguate with a synonym
            else:
                for s in synonyms:
                    if s != name:
                        name = f"{name} ({s})"
                        break

            if name in disambiguated_aliases:
                if disambiguated_aliases[name] != cui:
                        failed += 1
                        alt_ids2cui[cui] = disambiguated_aliases[name]
                        print(f"Failed to disambiguate:\nEntity 1. {cui}\nName: {name}\nMeta:{entity}\nEntity 2. {disambiguated_aliases[name]}\nMeta: {kb[disambiguated_aliases[name]]}\n\n")
            else:
                disambiguated_aliases[name] = cui

    print(f"Total homonyms resolved: {homonyms}")
    print(f"Failed to resolve {failed} homonyms :(")
    print(f"Final unique names: {len(disambiguated_aliases)}")
    print(f"Avg. names per cui: {sum(avg_names)/len(avg_names):.2f}" )

    statistics = {
            "num_cuis": len(kb),
            "num_names": len(disambiguated_aliases),
            "num_homonyms":f"{homonyms} ({homonyms/len(disambiguated_aliases):.2f})",
            "num_failed": f"{failed} ({failed/len(disambiguated_aliases):.2f})",
            "avg_names_x_cui": f"{sum(avg_names)/len(avg_names):.2f}"
        }
    return disambiguated_aliases, alt_ids2cui, statistics


def parse_args():
    parser = argparse.ArgumentParser(description="Inject obo ontology from URL or local file")
    parser.add_argument("--ontology_path", type=str, required=True,
                        help="Path to .obo file (e.g., ../datasets/ncbitaxon.obo) or a URL")
    parser.add_argument("--output_dir", type=str,
                        default="obo_ontology", 
                        help="Path of output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    filepath = args.ontology_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kb = build_kb(filepath)
    name2cui, new_kb, alt_ids2cui = get_name2cuis_mapping(kb)
    disambiguated_names, alt_ids2cui, statistics = resolve_homonyms(name2cui, new_kb, alt_ids2cui)

    excluded_cuis = ["HP:0000001"]
    disambiguated_kb = []
    for name, cui in disambiguated_names.items():
        if name.strip() and cui.strip():
            clean_name = re.sub(r"\n", " ", name.strip())
        if cui not in excluded_cuis:
            disambiguated_kb.append({"id": cui.strip(), "name": clean_name})

    with gzip.open(output_dir /"ontology.json.gz", "wt", encoding="utf-8") as file:
        json.dump(disambiguated_kb, file, indent=4)

    with open(output_dir / "alt_ids2cui.json", "w") as file:
        json.dump(alt_ids2cui, file, indent=4)
    
    with open(output_dir / "statistics_kb.json", "w") as file:
        json.dump(statistics, file, indent=4)

if __name__ == "__main__":
    main()