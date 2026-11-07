import re

# (loan_type, status)
META = {
    "eligibility.md":          ("personal", "current"),
    "prepayment.md":           ("personal", "current"),
    "home_loan_eligibility.md": ("home", "current"),
    "home_loan_prepayment.md":  ("home", "current"),
    "collections.md":          ("general", "superseded"),
    "collections_v2.md":       ("general", "current"),
    "interest_rates.md":       ("general", "current"),
    "loan_limits.md":          ("general", "current"),
    "kyc_documents.md":        ("general", "current"),
    "grievance.md":            ("general", "current"),
}
OLD = re.compile(r"\b(older|previous|prior|old|earlier|superseded|v1|version 1(\.0)?)\b", re.I)

def query_allowed_sources(q, sources):
    has_p = bool(re.search(r"\bpersonal\b", q, re.I))
    has_h = bool(re.search(r"\bhome\b", q, re.I))
    types = None
    if has_p and not has_h:
        types = {"personal", "general"}
    elif has_h and not has_p:
        types = {"home", "general"}
    current_only = not OLD.search(q)
    out = []
    for s in sources:
        t, st = META.get(s, ("general", "current"))
        if types is not None and t not in types:
            continue
        if current_only and st != "current":
            continue
        out.append(s)
    return out or None
