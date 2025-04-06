class AgentState:
    """Keep track of query vibes for quick repeats."""
    def __init__(self):
        self.cache = {}

    def get_response(self, query):
        """Check if we’ve seen this query before."""
        return self.cache.get(query.lower())

    def save_response(self, query, sql):
        """Stash the SQL for later."""
        self.cache[query.lower()] = sql