SRC = "1706.03762v7.pdf"

EVAL_QUESTIONS = [
    # ---------- Specific fact lookups (body text) ----------
    {
        "question": "How many attention heads does the base Transformer use, and what is the dimension of each head?",
        "expected_source": SRC,
        "expected_answer_contains": ["8", "64"],
    },
    {
        "question": "What is the dimensionality of the inner layer of the position-wise feed-forward network?",
        "expected_source": SRC,
        "expected_answer_contains": ["2048"],
    },
    {
        "question": "What optimizer was used for training, and how many warmup steps were used?",
        "expected_source": SRC,
        "expected_answer_contains": ["Adam", "4000"],
    },
    {
        "question": "How long were the big models trained, and on what hardware?",
        "expected_source": SRC,
        "expected_answer_contains": ["3.5", "P100"],
    },
    {
        "question": "About how many sentence pairs are in the WMT 2014 English-German training dataset?",
        "expected_source": SRC,
        "expected_answer_contains": ["4.5"],
    },
    {
        "question": "What length penalty was used with beam search during translation?",
        "expected_source": SRC,
        "expected_answer_contains": ["0.6"],
    },

    # ---------- Buried facts (mid-document sections) ----------
    {
        "question": "How much worse than the best setting was single-head attention, in BLEU?",
        "expected_source": SRC,
        "expected_answer_contains": ["0.9"],
    },

    # ---------- Spans multiple sections ----------
    {
        "question": "Compare self-attention and recurrent layers in sequential operations and maximum path length.",
        "expected_source": SRC,
        "expected_answer_contains": ["O(1)", "O(n)"],
    },

    # ---------- Table-dependent (scored separately) ----------
    {
        "question": "What F1 score did the 4-layer Transformer achieve in the semi-supervised setting on WSJ section 23?",
        "expected_source": SRC,
        "expected_answer_contains": ["92.7"],
        "needs": "table",  # Table 4
    },
    {
        "question": "What was the training cost in FLOPs of the Transformer base model for English-to-German?",
        "expected_source": SRC,
        "expected_answer_contains": ["3.3"],
        "needs": "table",  # Table 2 (exponent 10^18 gets flattened to "1018")
    },
    {
        "question": "What was the dev perplexity (PPL) of the configuration with a single attention head?",
        "expected_source": SRC,
        "expected_answer_contains": ["5.29"],
        "needs": "table",  # Table 3 row (A), needs column alignment
    },
    {
        "question": "How many parameters (in millions) did the big model have?",
        "expected_source": SRC,
        "expected_answer_contains": ["213"],
        "needs": "table",  # Table 3 last row
    },

    # ---------- Known conflict inside the paper itself ----------
    {
        "question": "What BLEU score did the big Transformer achieve on WMT 2014 English-to-French?",
        "expected_source": SRC,
        "expected_answer_contains": ["41.8"],
        "note": "Abstract and Table 2 say 41.8, but Section 6.1 says 41.0. A 41.0 answer is a faithful quote, not a retrieval bug.",
    },

    # ---------- Out of scope: should refuse ----------
    {
        "question": "How many parameters does GPT-3 have?",
        "expected_source": None,
        "expected_answer_contains": None,
    },
    {
        "question": "What BLEU score did the Transformer get on English-to-Spanish translation?",
        "expected_source": None,
        "expected_answer_contains": None,
    },
    {
        "question": "What was the total dollar cost of training the Transformer?",
        "expected_source": None,  # paper reports FLOPs, never dollars
        "expected_answer_contains": None,
    },
]