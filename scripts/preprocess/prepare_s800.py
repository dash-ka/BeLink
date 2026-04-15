from bioc import pubtator, biocxml
import gzip, argparse, os, json, re, bioc, copy
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict
from entitytools.entitytools.file_formats import pubtator_to_bioc, save_bioc_docs
from utils import mark_sentences, load_cui_set, filter_unseen_queries

def filter_collection(args, collection):

    oov = set()
    counter = 0
    skipped_non_contiguous = 0

    print("Loading dictionary...")
    cui_set = load_cui_set(os.path.join(args.terminology_dir, "terminology.json.gz"))
    print("CUI set size: ", len(cui_set))

    with open(os.path.join(args.terminology_dir, "alt_ids2cui.json")) as f:
        alt_ids2cui = json.load(f)

    for doc in collection:
        for passage in doc.passages:
            filtered_annotations = []
            for anno in passage.annotations:

                raw_cui = anno.infons.get("concept_id", "").strip()
                # Normalize separators to commas
                normalized_cui = raw_cui.strip("-")
                if any(map(lambda x: x in normalized_cui, ["|", "+", ",", ";"])):
                    continue

                for sep in (";", "-", "|", ",", "+"):
                    normalized_cui = normalized_cui.replace(sep, ",")
                if not normalized_cui:
                    continue                

                # Split and remove taxonomy suffix
                #cuis = [c.split("(Tax:")[0].strip() for c in normalized_cui.split(",") if c.strip()]
                cuis = [c.lower().split("(tax:")[0] for c in normalized_cui.split(",")]
                if len(cuis)>1:
                    skipped_non_contiguous +=1
                    continue

                cui = alt_ids2cui.get(cuis[0], cuis[0])
                
                if cui in cui_set:
                    anno.infons["concept_id"] = cui
                    anno.text = anno.text.strip().replace("\n", " ")
                    filtered_annotations.append(anno)
                else:
                    print(cui, anno.infons["concept_id"])
                    oov.add(anno.infons["concept_id"])

            passage.annotations = filtered_annotations
            counter += len(filtered_annotations)

    print(f"Number of queries: {counter}")
    print(f"OOV set size: {len(oov)}")
    print("Skipped non contiguous mentions: ", skipped_non_contiguous)

    return collection

def main():
    parser = argparse.ArgumentParser(description='Convert S800 to BioCXML')
    parser.add_argument('--data_dir',required=True,type=str,help='Path to the Corpus data.')
    parser.add_argument('--terminology_dir',required=True,type=str,help='Path to disambiguated terminology.')
    parser.add_argument('--output_dir',required=True,type=str,help='Path to the folder where to store the Gzipped BioC XML files')
    parser.add_argument('--filter_test',required=False, action="store_true",
                        help="If specified, filter test mentions")
    parser.add_argument('--with_sentences',required=False, action="store_true",
                        help="If specified, mark sentneces in collections")
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    XML_INVALID_CHARS = re.compile(
        r'[\x00-\x08\x0B\x0C\x0E-\x1F]'
    )
    def sanitize_xml_text(text):
        if text is None:
            return ""
        return XML_INVALID_CHARS.sub('', text)
    
    train_docs, test_docs = [], []
    print("Processing the corpus:")
    with open(os.path.join(args.data_dir, "Corpus/S800.tsv")) as file:
        lines = file.readlines()

    data = defaultdict(list)
    for line in lines:
        if line:
            cui, pmid_raw, start, end, mention = line.split("\t")
            pmid = pmid_raw.split(":")[0]
            data[pmid].append({
                "cui":cui, 
                "mention": mention.strip(), 
                "start": int(start), 
                "end": int(end)+1
                })
            
    for idx, pmid in tqdm(enumerate(data), total=len(data)):

        # collect context
        filepath = os.path.join(args.data_dir, "Corpus/abstracts", pmid+".txt")
        with open(filepath) as fin:
            context = fin.read()

        bioc_doc = bioc.BioCDocument()
        bioc_doc.id = pmid
        bioc_passage = bioc.BioCPassage()
        bioc_passage.text = sanitize_xml_text(context)
        bioc_passage.offset = 0
        bioc_doc.add_passage(bioc_passage)

        for entry in data[pmid]:

            if any(map(lambda x: x in entry["cui"], ["|", "+", ",", ";"])):
                continue

            bioc_anno = bioc.BioCAnnotation()
            bioc_anno.infons['concept_id'] = entry["cui"]
            bioc_anno.text = entry["mention"]
            start, end = entry["start"], entry["end"]
            bioc_loc = bioc.BioCLocation(start,end-start)
            bioc_anno.add_location(bioc_loc)
            bioc_passage.add_annotation(bioc_anno)
            
        if idx < 500:
            train_docs.append(bioc_doc)
        else:
            test_docs.append(bioc_doc)
        
    print("Filtering annotations for those in ontology...")
    train_docs = filter_collection(args, train_docs)
    test_docs = filter_collection(args, test_docs)
    
    if args.filter_test:
        unseen_queries = filter_unseen_queries(test_docs, [train_docs])
        mark_sentences(unseen_queries)
        save_bioc_docs(unseen_queries, output_dir / "unseen_test.bioc.xml.gz")
        
    if args.with_sentences:
        print("Marking sentences...")
        mark_sentences(train_docs)
        mark_sentences(test_docs)

    print("Saving S800 documents...")

    save_bioc_docs(train_docs, output_dir / "traindev.bioc.xml.gz")
    print(f"Train docs:{len(train_docs)}\nTest docs: {len(test_docs)}")
    print("Done")
	
if __name__ == '__main__':
	main()


