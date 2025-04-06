def log_feedback(query, is_correct):
    """Drop feedback into a chill log file."""
    with open("feedback_log.txt", "a") as f:
        f.write(f"Query: {query}, Correct: {is_correct}\n") 