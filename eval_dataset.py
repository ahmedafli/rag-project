EVAL_QUESTIONS = [
    {
        "question": "What percentage of Denmark's electricity comes from wind power?",
        "expected_source": "sample_rag_notes.pdf",
        "expected_answer_contains": ["50%"],
    },
    {
        "question": "How efficient are modern commercial solar panels?",
        "expected_source": "sample_rag_notes.pdf",
        "expected_answer_contains": ["20", "22"],
    },
    {
        "question": "How much power can the Three Gorges Dam generate?",
        "expected_source": "sample_rag_notes.pdf",
        "expected_answer_contains": ["20,000", "megawatts"],
    },
    {
        "question": "How has grid-scale battery storage cost changed since 2010?",
        "expected_source": "sample_rag_notes.pdf",
        "expected_answer_contains": ["90%"],
    },
    {
        "question": "What is the price of a Tesla Powerwall?",
        "expected_source": None,
        "expected_answer_contains": None,
    },
]