from bioc import pubtator, biocxml
import gzip, argparse, os, csv, json, re, bioc, copy
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
from pathlib import Path
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

    with open(os.path.join(args.dictionary_dir, "processed_kb.json"), encoding="utf-8") as f:
        kb = json.load(f)

    for doc in collection:
        for passage in doc.passages:
            filtered_annotations = []
            for anno in passage.annotations:

                raw_cui = anno.infons.get("concept_id", "").strip()
                # Normalize separators to commas
                normalized_cui = raw_cui.strip("-")
                for sep in (";", "-"):
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
                    taxon = kb[cui]["organism"]
                    anno.infons["concept_id"] = cui
                    anno.text = anno.text.strip().replace("\n", " ")
                    if args.qualify:
                        anno.text = f"{anno.text} ({taxon})"
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
    parser = argparse.ArgumentParser(description='Convert GNormPLus to BioCXML')
    parser.add_argument('--data_dir',required=True,type=str,help='Directory with source corpus files')
    parser.add_argument('--terminology_dir',required=True,type=str,help='Path to the disambiguated terminology.')
    parser.add_argument('--output_dir',required=True,type=str,help='Path to the folder where to store Gzipped BioC XML files')
    parser.add_argument('--qualify',required=False, action="store_true",
                        help="If specified, qualify mentions with organism name")
    parser.add_argument('--filter_test',required=False, action="store_true",
                        help="If specified, filter test mentions")
    parser.add_argument('--with_sentences',required=False, action="store_true",
                        help="If specified, mark sentences in collections")
    
    args = parser.parse_args()
    assert os.path.isdir(args.data_dir)

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading GNormPlus documents...")
    with open(f"{args.data_dir}/BC2GNtrain.PubTator.txt") as fp:
        gnormplus_train_docs = pubtator.load(fp)
    with open(f"{args.data_dir}/BC2GNtest.PubTator.txt") as fp:
        gnormplus_test_docs = pubtator.load(fp)

    gnormplus_train_docs = [ pubtator_to_bioc(doc) for doc in gnormplus_train_docs ]
    gnormplus_test_docs = [ pubtator_to_bioc(doc) for doc in gnormplus_test_docs ] 

    print("Filtering annotations for those in ontology...")
    gnormplus_train_docs = filter_collection(args, gnormplus_train_docs)
    gnormplus_test_docs = filter_collection(args, gnormplus_test_docs)

    if args.filter_test:
        unseen_queries = filter_unseen_queries(gnormplus_test_docs, [gnormplus_train_docs])
        mark_sentences(unseen_queries)
        save_bioc_docs(unseen_queries, output_dir / "unseen_test.bioc.xml.gz")
        
    if args.with_sentences:
        print("Marking sentences...")
        mark_sentences(gnormplus_train_docs)
        #mark_sentences(gnormplus_test_docs)

    print("Saving documents GNormPlus Gene...")
    if args.qualify:
        train_file = "traindev_qualified.bioc.xml.gz"
        test_file = "test_qualified.bioc.xml.gz"
    else:
        train_file = "traindev.bioc.xml.gz"
        test_file = "test.bioc.xml.gz"

    save_bioc_docs(gnormplus_train_docs, output_dir / train_file)

    print(f"Train docs:{len(gnormplus_train_docs)}\nTest docs: {len(gnormplus_test_docs)}")
    print("Done")
	
if __name__ == '__main__':
	main()


