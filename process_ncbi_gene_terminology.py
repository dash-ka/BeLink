#!/usr/bin/env python

import argparse
import gzip, re, json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

def build_kb(filepath: list, human_only=False, qualify=False) -> dict:
    """Parse the content into a structured KB dictionary."""

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.readlines()
    
    kb = dict()
    taxon2gene = defaultdict(int)
    for line in content[1:]:
        fields = line.split("\t")
        if len(fields) < 17:
            print("Skipped incomplete row:", fields)
            continue
        
        if human_only and ("homo sapiens" not in fields[1].lower()):
            continue

        gene_symbol = fields[5].strip()
        description = fields[7].strip()
        organism = fields[1].strip()
        cui = fields[2].strip()
        synonyms_raw = fields[6].strip()
        pref_name = description if description else gene_symbol 
        synonyms = "|".join(set(synonyms_raw.split(", "))) if synonyms_raw else "" 
        
        if qualify: 
            synonyms = "|".join([f"{s} ({organism})" for s in set(synonyms_raw.split(", "))]) if fields[6].strip() else "" 
            pref_name = f"{pref_name} ({organism})" 
            gene_symbol = f"{gene_symbol} ({organism})" 
        
        kb[cui] = {
                "cui": cui,
                "name": pref_name,
                "synonyms": gene_symbol + "|" + synonyms if synonyms else gene_symbol,
                "organism": organism
            }
        taxon2gene[organism] +=1

    print(f"KB loaded with {len(kb)} entries.")
    return kb, taxon2gene

def get_name2cuis_mapping(kb: dict) -> tuple:

    alt_ids2cui, new_kb = dict(), dict()
    name2cui = defaultdict(list)
    pattern = re.compile("[+;,]")

    for cui, entry in tqdm(kb.items(), total=len(kb), desc="Processing KB"):
        
        alt_cuis = []

        # Normalize CUI by removing composite identifiers
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

        new_kb[cui] = {"name":pref_name, "synonyms":synonyms, "organism":entry["organism"]}

    print(f"Old KB #entities: {len(kb)}\nNew KB # entities: {len(new_kb)}" )
    return name2cui, new_kb, alt_ids2cui

def resolve_homonyms(name2cui: list, kb: dict, alt_ids2cui: dict, qualified=False) -> dict:
    
    """Resolve duplicate term names mapping to different CUIs."""
    
    homonyms, failed = 0, 0
    avg_names = []
    is_homonym = lambda x: len(set(name2cui.get(x, []))) > 1
    disambiguated_aliases = dict()

    for cui, entity in tqdm(kb.items(), desc="Resolving homonyms"):
        cui = alt_ids2cui.get(cui, cui)
        pref_name = entity["name"]
        organism = entity["organism"]
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
                        if synonyms:
                            for s in synonyms:
                                if s not in [name, pref_name]:
                                    new_name = f"{name} ({s})"
                                    #print(f"Fallback disambiguation with synonym: {new_name}")
                                    break
                        if is_homonym(new_name) and not qualified:
                            new_name = f"{new_name} ({organism})"
                            #print(f"Fallback disambiguation with organism name: {name}")

                    name = new_name
                
                # If preferred name is the same, disambiguate with a synonym or organism
                else:
                    if synonyms:
                        for s in synonyms:
                            if s != name:
                                name = f"{name} ({s})"
                                #print(f"Fallback disambiguation with synonym: {name}")
                                break
                            
                    if is_homonym(new_name) and not qualified:
                        name = f"{name} ({organism})"
                        #print(f"Fallback disambiguation with organism name: {name}")

            if name in disambiguated_aliases:
                if disambiguated_aliases[name] != cui:
                    failed +=1
                    alt_ids2cui[cui] = disambiguated_aliases[name]
                    #print(f"{failed} Failed to disambiguate:\nEntity 1. {cui}\nName: {name}\nMeta:{entity}\nEntity 2. {disambiguated_aliases[name]}\nMeta: {kb[disambiguated_aliases[name]]}\n")

            else:            
                disambiguated_aliases[name] = cui


    print(f"Total homonyms: {homonyms}")
    print(f"Failed to resolve {failed} homonyms :(")
    print(f"Final unique names: {len(disambiguated_aliases)}")
    print(f"Avg. names per cui: {sum(avg_names)/len(avg_names):.2f}" )

    statistics = {
            "num_cuis": len(kb),
            "num_names": len(disambiguated_aliases),
            "num_homonyms":f"{homonyms} ({(homonyms/len(disambiguated_aliases))*100:.2f})",
            "num_failed": f"{failed} ({(failed/len(disambiguated_aliases))*100:.2f})",
            "avg_names_x_cui": f"{sum(avg_names)/len(avg_names):.2f}"
        }
    
    return disambiguated_aliases, alt_ids2cui, statistics


def parse_args():
    parser = argparse.ArgumentParser(description="Process NCBI Gene knowledge base.")
    parser.add_argument("--dictionary_path", type=str, required=True,
                        help="Path to the txt file")
    parser.add_argument("--output_dir", type=str,
                        default="../datasets/ncbi-gene-grf", 
                        help="Path of output directory")
    parser.add_argument("--human_only", action="store_true",
                        help="If specified, only include human genes")
    parser.add_argument("--qualify", action="store_true",
                        help="If specified, qualify genes with organism names ex: gene name (organism)")
    return parser.parse_args()

def main():
    args = parse_args()
    filepath = args.dictionary_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kb, taxon2gene = build_kb(filepath, human_only=args.human_only, qualify=args.qualify)
    name2cui, new_kb, alt_ids2cui = get_name2cuis_mapping(kb)
    disambiguated_aliases, alt_ids2cui, statistics = resolve_homonyms(name2cui, new_kb, alt_ids2cui, qualified=args.qualify)

    disambiguated_kb = []
    for name, cui in disambiguated_aliases.items():
        if name.strip() and cui.strip():
            clean_name = re.sub(r"\n", " ", name.strip())
        disambiguated_kb.append({"id": cui.strip(), "name": clean_name})

    with open(output_dir/"statistics_kb.json", "w") as file:
        json.dump(statistics, file, indent=4)

    with open(output_dir / "statistics_taxon.json", "w") as file:
        json.dump(taxon2gene, file, indent=4)

    with open(output_dir / "alt_ids2cui.json", "w") as file:
        json.dump(alt_ids2cui, file, indent=4)

    with open(output_dir / "processed_kb.json", "w") as file:
        json.dump(new_kb, file, indent=4)

    with gzip.open(output_dir / "terminology.json.gz", "wt", encoding="utf-8") as file:
        json.dump(disambiguated_kb, file, indent=4)

if __name__ == "__main__":
    main()