import re

from openai import OpenAI
from rag import RAGPipeline, build_prompt, generate_answer, rerank
from eval_dataset import EVAL_QUESTIONS
import config

REFUSAL_PHRASES = [
    "does not contain", "cannot answer", "no information",
    "not mentioned", "doesn't provide", "does not provide",
    "context doesn't", "context does not", "unable to answer",
]


def get_groq_client() -> OpenAI:
    return OpenAI(base_url=config.GROQ_BASE_URL, api_key=config.GROQ_API_KEY)


# ---------- Cheap, rule-based checks ----------

def check_retrieval_hit(sources: list[dict], expected_source: str | None) -> bool | None:
    """Did the expected document actually get retrieved?"""
    if expected_source is None:
        return None
    return any(s["source"] == expected_source for s in sources)


def check_keywords(answer: str, expected_keywords: list[str] | None) -> bool | None:
    """
    Exact-match check for facts that CAN'T be paraphrased -- numbers, %s,
    proper nouns. Good for catching wrong numbers, but blind to synonyms.
    """
    if expected_keywords is None:
        return None
    normalized_answer = re.sub(r"\s+", "", answer.lower())
    return all(re.sub(r"\s+", "", kw.lower()) in normalized_answer for kw in expected_keywords)


def llm_judge_refusal(groq, question: str, answer: str) -> bool:
    """Did the model correctly decline to answer, in substance, regardless of exact phrasing?"""
    judge_prompt = f"""Question: {question}
Answer given: {answer}

Does the answer correctly decline to answer because the information isn't available, rather than guessing or fabricating a response? Respond with ONLY one word: "YES" or "NO".
"""
    response = groq.chat.completions.create(
        model=config.JUDGE_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    verdict = (response.choices[0].message.content or "").strip().upper()
    return verdict.startswith("YES")


# ---------- LLM-as-judge checks (understand paraphrasing) ----------

def llm_judge_faithfulness(groq, context_chunks: list[dict], answer: str) -> bool:
    """Is every claim in the answer actually supported by the retrieved context?"""
    context_text = "\n\n".join(c["text"] for c in context_chunks)
    judge_prompt = f"""You are a strict fact-checker. Given a CONTEXT and an ANSWER, determine if the ANSWER is fully supported by the CONTEXT -- meaning every claim can be verified from the context, with no invented facts.

CONTEXT:
{context_text}

ANSWER:
{answer}

Respond with ONLY one word: "YES" if fully supported, "NO" otherwise.
"""
    response = groq.chat.completions.create(
        model=config.JUDGE_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    verdict = (response.choices[0].message.content or "").strip().upper()
    return verdict.startswith("YES")


def llm_judge_correctness(groq, question: str, answer: str, expected_facts: list[str] | None) -> bool | None:
    """
    Does the answer correctly convey the required facts, REGARDLESS of exact
    wording? Catches paraphrases ("decreased" vs "fallen") that exact keyword
    matching would wrongly mark as failures.
    """
    if expected_facts is None:
        return None
    facts_text = ", ".join(expected_facts)
    judge_prompt = f"""Question: {question}
Answer given: {answer}
Key facts that MUST be conveyed (in any wording): {facts_text}

Does the answer correctly convey all of these key facts, regardless of exact wording? Respond with ONLY one word: "YES" or "NO".
"""
    response = groq.chat.completions.create(
        model=config.JUDGE_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    verdict = (response.choices[0].message.content or "").strip().upper()
    return verdict.startswith("YES")


# ---------- Main eval loop ----------

def main():
    pipeline = RAGPipeline()
    groq = get_groq_client()

    results = []

    for i, item in enumerate(EVAL_QUESTIONS, start=1):
        question = item["question"]
        expected_source = item["expected_source"]
        expected_keywords = item["expected_answer_contains"]

        print("\n" + "=" * 80)
        print(f"QUESTION {i}: {question}")
        print("=" * 80)

        try:
            candidates = pipeline.hybrid_retrieve(question, fetch_k=10, final_k=10)
            chunks = rerank(question, candidates, top_n=config.TOP_K)
            prompt = build_prompt(question, chunks)
            answer = generate_answer(pipeline.nvidia, prompt)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "retrieval_hit": None,
                "keyword_pass": None,
                "refusal_correct": None,
                "faithful": None,
                "correct": None,
                "needs": item.get("needs"),
            })
            continue

        retrieval_hit = check_retrieval_hit(chunks, expected_source)
        keyword_pass = check_keywords(answer, expected_keywords)
        is_refusal_case = expected_source is None
        refusal_correct = llm_judge_refusal(groq, question, answer) if is_refusal_case else None
        faithful = llm_judge_faithfulness(groq, chunks, answer)
        correct = llm_judge_correctness(groq, question, answer, expected_keywords)

        print(f"\nAnswer: {answer}")
        print(f"\n  Retrieval hit:         {retrieval_hit}")
        print(f"  Keyword match (exact): {keyword_pass}")
        print(f"  LLM-judged correct:    {correct}")
        if is_refusal_case:
            print(f"  Correctly refused:     {refusal_correct}")
        print(f"  LLM-judged faithful:   {faithful}")

        results.append({
            "retrieval_hit": retrieval_hit,
            "keyword_pass": keyword_pass,
            "refusal_correct": refusal_correct,
            "faithful": faithful,
            "correct": correct,
            "needs": item.get("needs"),
        })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    def pct(lst):
        lst = [x for x in lst if x is not None]
        return f"{100 * sum(lst) / len(lst):.0f}%" if lst else "N/A"

    def counts(lst):
        lst = [x for x in lst if x is not None]
        return f"({sum(lst)}/{len(lst)})" if lst else ""

    metrics = {
        "Retrieval hit rate":       [r["retrieval_hit"] for r in results],
        "Keyword accuracy (exact)": [r["keyword_pass"] for r in results],
        "LLM-judged correctness":   [r["correct"] for r in results],
        "Correct refusals":         [r["refusal_correct"] for r in results],
        "LLM-judged faithfulness":  [r["faithful"] for r in results],
    }
    for label, values in metrics.items():
        print(f"{label:<28} {pct(values):>5}  {counts(values)}")

    table_results = [r for r in results if r["needs"] == "table"]
    if table_results:
        print("\nTABLE-DEPENDENT QUESTIONS ONLY")
        for label, key in [("Keyword accuracy (exact)", "keyword_pass"),
                           ("LLM-judged correctness", "correct")]:
            vals = [r[key] for r in table_results]
            print(f"{label:<28} {pct(vals):>5}  {counts(vals)}")


if __name__ == "__main__":
    main()