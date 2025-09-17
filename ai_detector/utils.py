import kss

def remove_outliers_iqr(probs):
    sorted_probs = sorted(probs)
    q1 = sorted_probs[len(sorted_probs) // 4]
    q3 = sorted_probs[len(sorted_probs) * 3 // 4]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered = [p for p in probs if lower <= p <= upper]
    return filtered

def chunk_text(text: str, tokenizer, max_len):
    chunks = []
    current_chunk_sentences = []
    current_length = 0
    max_len_for_chunking = max_len - 2

    sentences = kss.split_sentences(text)

    for sentence in sentences:
        sentence_token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(sentence_token_ids)

        if sentence_length > max_len_for_chunking:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            chunks.append(sentence)
            current_chunk_sentences = []
            current_length = 0
            continue

        if current_length + sentence_length > max_len_for_chunking:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_length = sentence_length
        else:
            current_chunk_sentences.append(sentence)
            current_length += sentence_length

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    
    return chunks

def format_detection_results(chunk_probabilities):
    if not chunk_probabilities:
        return {
            "average_probability": None,
            "max_probability": None,
            "chunk_probabilities": [],
            "chunk_count": 0
        }
    
    filtered_probs = remove_outliers_iqr(chunk_probabilities)
    
    if not filtered_probs:
        return {
            "average_probability": None,
            "max_probability": None,
            "chunk_probabilities": chunk_probabilities,
            "chunk_count": len(chunk_probabilities)
        }

    avg_prob = round(sum(filtered_probs) / len(filtered_probs), 4)
    max_prob = round(max(filtered_probs), 4)
    
    return {
        "average_probability": avg_prob,
        "max_probability": max_prob,
        "chunk_probabilities": chunk_probabilities,
        "chunk_count": len(chunk_probabilities)
    }