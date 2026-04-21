import bioc, copy, gzip
from tqdm.auto import tqdm
from bioc import biocxml

def save_bioc_docs(docs, filename):
    """
    Saves a set of BioC documents.
    :param docs: the set of documents.
    :param filename: the name of the file on which to save the BioC docs.
    """
    collection = bioc.BioCCollection.of_documents(*docs)
    with gzip.open(filename, 'wt', encoding='utf8') as f:
        biocxml.dump(collection, f)

def get_sentence_annotations(passage, sentence):
    """
    Obtains the annotations within a sentence.
    :param passage: the complete passage.
    :param sentence: a sentence within the passage.
    :return: the annotations for the sentence.
    """
    sentence_start = sentence.offset
    sentence_end = sentence.offset + int(sentence.infons['length'])
    sentence_text = passage.text[(sentence_start - passage.offset):(sentence_end - passage.offset)]

    annotations = [anno for anno in passage.annotations if
                   anno.total_span.offset >= sentence_start and anno.total_span.end <= sentence_end]
    
    return annotations
    

def mark_sentences(collection):
    """Splits passage into sentences, and re-annotates at the sentence level."""
    
    import spacy
    nlp = spacy.load("en_core_web_sm")

    for doc in tqdm(collection):
        for passage in doc.passages:
            # We keep a copy of the original annotations to filter from
            # because we will eventually clear passage.annotations
            original_annotations = list(passage.annotations)
            
            # Temporary storage to see which annotations we've already "used"
            assigned_annotations = set()                

            if len(passage.annotations) > 0:
                i = 0
                current_anno = passage.annotations[i]
                start_offset = current_anno.locations[0].offset
                end_offset = start_offset + current_anno.locations[0].length
            else:
                start_offset = 10000000
                end_offset = 10000000

            parsed = nlp(passage.text)
            true_start = -1

            for sent in parsed.sents:
                start = sent[0].idx
                end = sent[-1].idx + len(sent[-1].text)
                while (end_offset - passage.offset) <= end and (i + 1) < len(passage.annotations):
                    i += 1
                    current_anno = passage.annotations[i]
                    start_offset = current_anno.locations[0].offset
                    end_offset = start_offset + current_anno.locations[0].length

                if (start_offset - passage.offset) > end or (end_offset - passage.offset) < end:
                    start = true_start if true_start != -1 else start
                    sentence = bioc.BioCSentence()
                    sentence.text = passage.text[start:end]
                    sentence.offset = passage.offset + start
                    sentence.infons['length'] = str(end - start)

                    # --- NEW LOGIC START ---
                    # Get annotations belonging to THIS specific sentence
                    matched_annos = get_sentence_annotations(passage, sentence)
                    
                    for anno in matched_annos:
                        sentence.add_annotation(anno)
                        assigned_annotations.add(anno)
                    # --- NEW LOGIC END ---

                    passage.add_sentence(sentence)
                    true_start = -1
                elif true_start == -1:
                    true_start = start
            if true_start != -1:
                sentence = bioc.BioCSentence()
                sentence.text = passage.text[true_start:len(passage.text)]
                sentence.offset = passage.offset + true_start
                sentence.infons['length'] = str(len(passage.text) - true_start)

                # Get annotations belonging to THIS specific sentence
                matched_annos = get_sentence_annotations(passage, sentence)
                
                for anno in matched_annos:
                    sentence.add_annotation(anno)
                    assigned_annotations.add(anno)
                # --- NEW LOGIC END ---
                passage.add_sentence(sentence)
                true_start = -1

            # Optional: Clean up passage level annotations 
            # (Removes them from passage so they only exist in sentences)
            passage.annotations = [a for a in original_annotations if a not in assigned_annotations]
            assert len(passage.annotations) <1

def load_cui_set(path):
        cui_set = set()
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                cui, *_ = line.strip().split('||')
                for c in cui.split('|'):
                    c = c.strip()
                    cui_set.add(c)
        return cui_set

def filter_unseen_queries(test_collection, train_collections_lst):

    seen_queries = 0
    traindev_ids, traindev_texts = set(), set()
    for seen_collection in train_collections_lst:
        for doc in seen_collection:
            for passage in doc.passages:
                for anno in passage.annotations:
                    traindev_ids.add((anno.infons['concept_id'], anno.text.lower())),
                    traindev_texts.add(anno.text.lower())
                    seen_queries += 1

    test_annotations = []
    unseen_test_collection = copy.deepcopy(test_collection)
    for doc in unseen_test_collection:
        for passage in doc.passages:
            filtered_annotations = []
            for anno in passage.annotations:
                if anno.text.lower() in traindev_texts:
                    continue
                test_annotations.append((anno.infons['concept_id'], anno.text.lower()))
                filtered_annotations.append(anno)
            passage.annotations = filtered_annotations
        
    print("# of unseen queries:", len(set(test_annotations)))
    print("# of seen queries:", seen_queries)
    return unseen_test_collection
