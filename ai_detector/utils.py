def remove_outliers_iqr(probs):
    sorted_probs = sorted(probs)
    q1 = sorted_probs[len(sorted_probs) // 4]
    q3 = sorted_probs[len(sorted_probs) * 3 // 4]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered = [p for p in probs if lower <= p <= upper]
    return filtered

def format_detection_results(chunk_probabilities):
    if not chunk_probabilities:
        return {
            "average_probability": None,
            "max_probability": None,
        }
    
    filtered_probs = remove_outliers_iqr(chunk_probabilities)
    
    if not filtered_probs:
        return {
            "average_probability": None,
            "max_probability": None
            # "chunk_probabilities": chunk_probabilities,
            # "chunk_count": len(chunk_probabilities)
        }

    avg_prob = round(sum(filtered_probs) / len(filtered_probs), 4)
    max_prob = round(max(filtered_probs), 4)
    
    return {
        "average_probability": avg_prob,
        "max_probability": max_prob,
    }