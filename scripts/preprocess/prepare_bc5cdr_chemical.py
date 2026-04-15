from bioc import biocxml
import gzip, argparse,os,json, bioc, copy
from tqdm import tqdm
import xml.etree.ElementTree as ET
from pathlib import Path
from entitytools.entitytools.file_formats import save_bioc_docs
from utils import mark_sentences, load_cui_set, filter_unseen_queries

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

    print("Loading BC5CDR documents...")
    with open(f'{args.data_dir}/CDR_TrainingSet.BioC.xml') as f:
        bc5cdr_train_collection = biocxml.load(f)
    with open(f'{args.data_dir}/CDR_DevelopmentSet.BioC.xml') as f:
        bc5cdr_val_collection = biocxml.load(f)
    with open(f'{args.data_dir}/CDR_TestSet.BioC.xml') as f:
        bc5cdr_test_collection = biocxml.load(f)

    print("Reformatting BioC XML annotations for BC5CDR...")
    # remmember that bc5cdr contains both 'Disease' and 'Chemical' identifiers, thus oov are numerous
    for doc in bc5cdr_train_collection.documents + bc5cdr_val_collection.documents + bc5cdr_test_collection.documents:
        for passage in doc.passages:
            passage.annotations = [ 
                anno for anno in passage.annotations if \
                    (anno.infons['MESH'] != '-1') and
                    ('|' not in anno.infons['MESH']) and 
                    ("CompositeRole" not in anno.infons) and 
                    (anno.infons['type'] == 'Chemical') 
            ]
            for anno in passage.annotations:
                anno.infons = { 'concept_id': f"MESH:{anno.infons['MESH']}" }


    print("Filtering annotations for those in ontology...")
    train_docs_bc5cdr= filter_collections(args, bc5cdr_train_collection.documents)
    val_docs_bc5cdr = filter_collections(args, bc5cdr_val_collection.documents)
    test_docs_bc5cdr = filter_collections(args, bc5cdr_test_collection.documents)
    
    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.filter_test:
        unseen_queries = filter_unseen_queries(test_docs_bc5cdr, [train_docs_bc5cdr, val_docs_bc5cdr])
        mark_sentences(unseen_queries)
        save_bioc_docs(unseen_queries, output_dir / "unseen_test.bioc.xml.gz")
        
    if args.with_sentences:
        print("Marking sentences...")
        mark_sentences(train_docs_bc5cdr)
        mark_sentences(val_docs_bc5cdr)
        #mark_sentences(test_docs_bc5cdr)

    print("Saving documents BC5CDR Chemical ...")
    #save_bioc_docs(test_docs_bc5cdr, output_dir / "test.bioc.xml.gz")
    save_bioc_docs(train_docs_bc5cdr + val_docs_bc5cdr, output_dir / "traindev.bioc.xml.gz" )
    print(f"{len(train_docs_bc5cdr)=} {len(val_docs_bc5cdr)=} {len(train_docs_bc5cdr + val_docs_bc5cdr)=} {len(test_docs_bc5cdr)=}")

    print("Done")
	
if __name__ == '__main__':
	main()


