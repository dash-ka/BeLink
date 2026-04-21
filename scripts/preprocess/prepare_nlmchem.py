from bioc import biocxml
from pathlib import Path
import gzip, argparse, os, json
from belink.preprocess.utils import mark_sentences, load_cui_set, filter_unseen_queries, save_bioc_docs

def filter_collections(args, collection):

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
                    
                    cui = anno.infons["concept_id"].strip()
                    if any(map(lambda x: cui.startswith(x), ["C", "D"])):
                        cui = "MESH:" + cui
                    elif ":" not in cui:
                        cui = "OMIM:" + cui

                    new_id = alt_ids2cui.get(cui, cui)
                    if len(anno.infons)>1:
                        skipped_non_contiguous +=1
                        continue

                    if new_id in cui_set:
                        # over-write concept_id
                        anno.infons["concept_id"] = new_id
                        anno.text = anno.text.strip().replace("\n", " ")
                        filtered_annotations.append(anno)
                    else:
                        print(anno.infons["concept_id"])
                        oov.add(anno.infons["concept_id"])

                passage.annotations = filtered_annotations
                counter += len(filtered_annotations)
                
        print(f"Number of queries: {counter}")
        print(f"OOV set size: {len(oov)}")
        print("Skipped non contiguous mentions: ", skipped_non_contiguous)
        return collection

def main():
    parser = argparse.ArgumentParser(description='Convert NLM-Chem corpus to BioCXML and set up a matching MeSH ontology')
    parser.add_argument('--data_dir', required=True,type=str,help='Directory with source corpus files')
    parser.add_argument('--terminology_dir',required=True,type=str,help='Path to the folder with disambiguated terminology')
    parser.add_argument('--output_dir',required=True,type=str,help='Where to save Gzipped BioC XML files')
    parser.add_argument('--filter_test',required=False,action="store_true",help='Whether to filter test set to retain only mentions unseen during training')
    parser.add_argument('--with_sentences',required=False,action="store_true",help='Whether to split text into sentences.')
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir)

    print("Loading NLM-Chemical data...")
    with open(f'{args.data_dir}/pmcids_train.txt', encoding="utf-8") as f:
        train_pmcids = [ line.strip() for line in f ]
    with open(f'{args.data_dir}/pmcids_dev.txt') as f:
        val_pmcids = [ line.strip() for line in f ]
    with open(f'{args.data_dir}/pmcids_test.txt') as f:
        test_pmcids = [ line.strip() for line in f ]

    train_docs_nlm, val_docs_nlm, test_docs_nlm = [], [], []
    for filename in os.listdir(f"{args.data_dir}/ALL"):
        with open(f"{args.data_dir}/ALL/{filename}", encoding="utf-8") as fp:
            collection = biocxml.load(fp)
            for doc in collection.documents:
                if doc.id in train_pmcids:
                    train_docs_nlm.append(doc)
                elif doc.id in val_pmcids:
                    val_docs_nlm.append(doc)
                elif doc.id in test_pmcids:
                    test_docs_nlm.append(doc)
                else:
                    raise RuntimeError(f"{doc.id=} is not in one of the train/val/test groupings")

    print("Reformatting BioC XML annotations...")
    for doc in train_docs_nlm+val_docs_nlm+test_docs_nlm :
        for passage in doc.passages:
            passage.annotations = [ anno for anno in passage.annotations if
                                    (anno.infons['type'] == 'Chemical')  
                                    and ("CompositeRole" not in anno.infons) 
                                ]
            for anno in passage.annotations:
                anno.infons = { 'concept_id': anno.infons['identifier'] }


    print("Filtering annotations for those in ontology...")
    train_docs_nlm = filter_collections(args, train_docs_nlm)
    val_docs_nlm = filter_collections(args, val_docs_nlm)
    test_docs_nlm =  filter_collections(args, test_docs_nlm)

    
    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.filter_test:
        unseen_queries = filter_unseen_queries(test_docs_nlm, [train_docs_nlm, val_docs_nlm])
        mark_sentences(unseen_queries)
        save_bioc_docs(unseen_queries, output_dir / "unseen_test.bioc.xml.gz")
        
    if args.with_sentences:
        print("Marking sentences...")
        mark_sentences(train_docs_nlm)
        mark_sentences(val_docs_nlm)
        mark_sentences(test_docs_nlm)
        
    print("Saving documents NLM-Chem...")
    #save_bioc_docs(train_docs_nlm, output_dir / "train.bioc.xml.gz")
    #save_bioc_docs(val_docs_nlm, output_dir / "val.bioc.xml.gz")
    save_bioc_docs(train_docs_nlm+val_docs_nlm, output_dir / "traindev.bioc.xml.gz")
    print(f"{len(train_docs_nlm)=} {len(val_docs_nlm)=} {len(train_docs_nlm+val_docs_nlm)=} {len(test_docs_nlm)=}")

    print("Done")
	
if __name__ == '__main__':
	main()


