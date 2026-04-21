from bioc import pubtator
import gzip, argparse,os,json, bioc, copy
from tqdm import tqdm
import xml.etree.ElementTree as ET
from pathlib import Path
from belink.preprocess.utils import mark_sentences, load_cui_set, filter_unseen_queries, save_bioc_docs

def pubtator_to_bioc(doc):
    """
    Function that transforms a PubTator document into BioC.
    :param doc: the PubTator document.
    :return: the BioC document.
    """
    bioc_doc = bioc.BioCDocument()
    bioc_doc.id = doc.pmid
    bioc_passage = bioc.BioCPassage()
    bioc_passage.text = doc.text
    bioc_passage.offset = 0
    bioc_doc.add_passage(bioc_passage)

    title = doc.text.split('\n')[0]
    bioc_doc.infons['title'] = title

    for a in doc.annotations:
        bioc_anno = bioc.BioCAnnotation()
        bioc_anno.infons['concept_id'] = a.id
        bioc_anno.text = a.text
        bioc_loc = bioc.BioCLocation(a.start,a.end-a.start)
        bioc_anno.add_location(bioc_loc)
        bioc_passage.add_annotation(bioc_anno)

    return bioc_doc


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
    parser = argparse.ArgumentParser(description='Convert NCBI-Disease corpus to BioCXML and set up a matching MeSH ontology')
    parser.add_argument('--data_dir', required=True,type=str,help='Directory with source corpus files')
    parser.add_argument('--terminology_dir',required=True,type=str,help='Path to the folder with disambiguated terminology')
    parser.add_argument('--output_dir',required=True,type=str,help='Where to save Gzipped BioC XML files')
    parser.add_argument('--filter_test',required=False,action="store_true",help='Whether to filter test set to retain only mentions unseen during training')
    parser.add_argument('--with_sentences',required=False,action="store_true",help='Whether to split text into sentences.')
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir)

    print("Loading NCBI documents...")
    with open(f"{args.data_dir}/NCBItrainset_corpus.txt") as fp:
        ncbi_train_docs = pubtator.load(fp)
    with open(f"{args.data_dir}/NCBIdevelopset_corpus.txt") as fp:
        ncbi_val_docs = pubtator.load(fp)
    with open(f"{args.data_dir}/NCBItestset_corpus.txt") as fp:
        ncbi_test_docs = pubtator.load(fp)

    train_docs = [ pubtator_to_bioc(doc) for doc in ncbi_train_docs ]
    val_docs = [ pubtator_to_bioc(doc) for doc in ncbi_val_docs ] 
    test_docs = [ pubtator_to_bioc(doc) for doc in ncbi_test_docs ] 

    print("Filtering annotations for those in ontology...")
    train_docs= filter_collections(args, train_docs)
    val_docs = filter_collections(args, val_docs)
    test_docs = filter_collections(args, test_docs )
    

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.filter_test:
        unseen_queries = filter_unseen_queries(test_docs, [train_docs, val_docs])
        mark_sentences(unseen_queries)
        save_bioc_docs(unseen_queries, output_dir / "unseen_test.bioc.xml.gz")
        
    if args.with_sentences:
        print("Marking sentences...")
        mark_sentences(train_docs)
        mark_sentences(val_docs)
        #mark_sentences(test_docs)

    print("Saving documents NCBI Disease ...")
    #save_bioc_docs(train_docs, output_dir / "train.bioc.xml.gz" )
    #save_bioc_docs(val_docs, output_dir / "val.bioc.xml.gz")
    #save_bioc_docs(test_docs, output_dir / "test.bioc.xml.gz")
    save_bioc_docs(train_docs + val_docs, output_dir / "traindev.bioc.xml.gz" )
    print(f"{len(train_docs)=} {len(val_docs)=} {len(train_docs + val_docs)=} {len(test_docs)=}")

    print("Done")
	
if __name__ == '__main__':
	main()


