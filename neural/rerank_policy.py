"""Select candidates using facts available to the input runtime."""

# Database and UserDatabase in quanpin/word_lattice.h.
DICTIONARY_SOURCES = (0, 1)
COMPARED = 9


def select_candidates(candidates):
    """Return candidates that answer the key, leader coverage, and the dictionary gate."""
    if not candidates:
        return [], False, False

    leader = candidates[0]
    width = len(leader["text"])

    def answers_key(candidate):
        value = candidate.get("answers_key")
        return len(candidate["text"]) == width if value is None else value

    selected = [candidate for candidate in candidates if answers_key(candidate)][:COMPARED]
    trusted = leader.get("trusted_dictionary_hit")
    if trusted is None:
        trusted = leader["source"] in DICTIONARY_SOURCES
    return selected, answers_key(leader), trusted
