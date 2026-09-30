import base64
import re
import time
import fitz  # PyMuPDF python library that opens and reads pdf 
from openai import OpenAI

import config


def get_vision_client() -> OpenAI:
    return OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
        timeout=120,
    )


def find_table_pages(pdf_path: str) -> list[dict]:
    """Scan every page for a 'Table N: ...' caption. Returns page number + caption."""
    doc = fitz.open(pdf_path)
    results = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        for match in re.finditer(r"(Table\s+\d+\s*:[^\n]*)", text):
            results.append({"page": page_num, "caption": match.group(1).strip()})
    doc.close()
    return results


def render_page_to_base64_png(pdf_path: str, page_num: int, dpi: int = 100) -> str:
    doc = fitz.open(pdf_path)
    pix = doc[page_num].get_pixmap(dpi=dpi)
    png_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(png_bytes).decode("utf-8")


def transcribe_table(client: OpenAI, image_b64: str, caption: str, max_retries: int = 6) -> str | None:
    """
    Returns the transcribed markdown, or None if every attempt failed or came
    back truncated. Callers MUST check for None -- this function does not
    raise, so a persistent failure doesn't crash the whole ingestion run.

    IMPORTANT: a non-empty `content` string is NOT the same as a successful
    response. A model can return partial, truncated content (finish_reason
    == "length") that looks like real data but silently drops rows. Both
    emptiness AND truncation are treated as failures here.
    """
    prompt = f"""This page contains a table with caption: "{caption}"

Transcribe ONLY this table into a clean markdown table. Rules:
- Include all rows and columns with correct headers, correctly aligned.
- Preserve exact numeric values, including scientific notation exponents
  written clearly (e.g. "3.3 x 10^18", not a flattened "3.3 1018").
- Leave a cell empty in the markdown if it's empty in the original.
- Include the caption as a heading above the table.
- Output ONLY the markdown table and its caption -- no commentary, no explanation.
"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.VISION_MODEL,
                max_tokens=12000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }],
            )
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            if content and finish_reason != "length":
                return content

            if content and finish_reason == "length":
                print(f"    Attempt {attempt + 1}/{max_retries}: TRUNCATED "
                      f"(finish_reason=length, {len(content)} chars so far). Retrying...")
            else:
                print(f"    Attempt {attempt + 1}/{max_retries}: empty content, "
                      f"finish_reason={finish_reason}")

        except Exception as e:
            print(f"    Attempt {attempt + 1}/{max_retries} raised: {e}")

        wait_time = 15 * (attempt + 1)
        print(f"    Waiting {wait_time}s before retry...")
        time.sleep(wait_time)

    return None


def extract_all_tables(pdf_path: str) -> list[dict]:
    client = get_vision_client()
    table_pages = find_table_pages(pdf_path)
    tables = []
    for tp in table_pages:
        print(f"  Transcribing table on page {tp['page'] + 1}: {tp['caption'][:60]}...")
        image_b64 = render_page_to_base64_png(pdf_path, tp["page"])
        print(f"    Image size: {len(image_b64) // 1024} KB (base64)")
        try:
            markdown = transcribe_table(client, image_b64, tp["caption"])
            print(f"    Done.")
        except Exception as e:
            print(f"    FAILED: {e}")
            markdown = f"[Table transcription failed for: {tp['caption']}]"
        tables.append({"text": markdown, "page": tp["page"] + 1, "caption": tp["caption"]})
    return tables