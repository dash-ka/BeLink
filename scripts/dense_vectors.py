import torch
from more_itertools import chunked
from tqdm.auto import tqdm
import faiss

def make_dense_vectors(model, tokenizer, texts):
    dense_vectors = []
    with torch.no_grad():
        for batch in chunked(tqdm(texts), 1000):
            tokenized = tokenizer(batch, truncation=True, max_length=512, padding=True, return_tensors='pt')
            outputs = model(input_ids=tokenized['input_ids'].to(model.device), attention_mask=tokenized['attention_mask'].to(model.device))
            cls_vectors = outputs.last_hidden_state[:,0,:]
            dense_vectors.append(cls_vectors.cpu())
    
    dense_vectors = torch.vstack(dense_vectors).numpy()
    
    return dense_vectors


def make_dense_vectors_with_grf(model, tokenizer, texts, rocchio=False, alpha=0.5):
      
    """
    texts: dict or iterable where texts[q] has keys {'mention', 'feedback'}
    """
    mention_texts, feedback_texts = zip(
            *[(texts[q]["mention"], texts[q]["feedback"]) for q in texts]
            )
    
    if not feedback_texts[0]: 
        dense_vectors = make_dense_vectors(model, tokenizer, mention_texts)
        print("No feedback available. Default to mention embedding without GRF!")
        
    else:        
        print(f"Applying GRF with alpha={alpha} and rocchio={rocchio}")

        mention_vectors = make_dense_vectors(model, tokenizer, mention_texts)
        feedback_vectors = make_dense_vectors(model, tokenizer, feedback_texts)
        assert mention_vectors.shape == feedback_vectors.shape
        if rocchio:
            # weighted interpolation
            dense_vectors = alpha * mention_vectors + (1 - alpha) * feedback_vectors
        else:
            # simple unweighted average
            dense_vectors = (mention_vectors + feedback_vectors) / 2

    return dense_vectors

def make_dense_lookup(model, tokenizer, onto_vectors, anno_texts, top_k, with_grf=False, use_rocchio=False, alpha=.5):
    
    if with_grf:
        query_vectors = make_dense_vectors_with_grf(model, tokenizer, anno_texts, rocchio=use_rocchio, alpha=alpha)
    else:
        query_vectors = make_dense_vectors(model, tokenizer, anno_texts)

    # indexing ontology vectors
    index_flat = faiss.IndexFlatIP(onto_vectors.shape[1])
    #gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    index_flat.add(onto_vectors)

    # candidate retrieval
    print(f"Retrieving {top_k} candidates for each query.")
    eval_biencoder_distances, eval_biencoder_indices = index_flat.search(query_vectors, top_k)

    lookup_by_mention_text = {}
    for i, anno_text in enumerate(anno_texts):
        #lookup_by_mention_text[anno_text] = {"candidates":eval_biencoder_indices[i].tolist()}
        # Convert indices to list of strings for XML compatibility
        candidates = [str(idx) for idx in eval_biencoder_indices[i].tolist()]
        lookup_by_mention_text[anno_text] = "|".join(candidates)

    return lookup_by_mention_text
    
