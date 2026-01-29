import math
from collections import Counter

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_cer(references, candidates):
    total_dist = 0
    total_len = 0
    
    for ref, cand in zip(references, candidates):
        dist = levenshtein_distance(ref, cand)
        total_dist += dist
        total_len += len(ref)
        
    if total_len == 0:
        return 0.0
        
    return total_dist / total_len

def count_ngrams(segment, n):
    return Counter([tuple(segment[i:i+n]) for i in range(len(segment)-n+1)])

def calculate_bleu(references, candidates, max_n=4):
    """
    Simplified BLEU score implementation.
    references: list of reference strings
    candidates: list of candidate strings
    """
    total_p = 0
    
    # Pre-tokenize (character level is already tokenized effectively, but let's treat string as list of chars)
    # Actually, for this task, "words" are characters? No, usually BLEU is word-level.
    # But the user asked for "Tokenization: choose proper tokenization strategy...".
    # Since I used char-level model, the output is characters.
    # For BLEU, we can treat characters as tokens (Char-BLEU) or reconstruct words and use Word-BLEU.
    # Given it's transliteration, Char-BLEU is often used, but standard BLEU is word-based.
    # However, since the target is Roman Urdu, space-separated words exist.
    # I will split by space for BLEU calculation to make it "Word-BLEU".
    
    ref_tokens = [r.split() for r in references]
    cand_tokens = [c.split() for c in candidates]
    
    precisions = []
    
    for n in range(1, max_n + 1):
        total_ngram_matches = 0
        total_candidate_ngrams = 0
        
        for ref, cand in zip(ref_tokens, cand_tokens):
            ref_ngrams = count_ngrams(ref, n)
            cand_ngrams = count_ngrams(cand, n)
            
            for ngram, count in cand_ngrams.items():
                total_ngram_matches += min(count, ref_ngrams[ngram])
            
            total_candidate_ngrams += sum(cand_ngrams.values())
            
        if total_candidate_ngrams == 0:
            precisions.append(0)
        else:
            precisions.append(total_ngram_matches / total_candidate_ngrams)
            
    if min(precisions) == 0:
        return 0.0
        
    geometric_mean = math.exp(sum([math.log(p) for p in precisions]) / max_n)
    
    # Brevity Penalty
    c = sum(len(cand) for cand in cand_tokens)
    r = sum(len(ref) for ref in ref_tokens)
    
    if c > r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c) if c > 0 else 0.0
        
    return bp * geometric_mean

def calculate_perplexity(loss):
    return math.exp(loss)
