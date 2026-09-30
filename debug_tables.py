from extract_tables import extract_all_tables

tables = extract_all_tables('data/1706.03762v7.pdf')

for i, t in enumerate(tables):
    text_type = type(t["text"])
    print(f"--- Table {i}: type={text_type} ---")
    if t["text"] is not None:
        print(t["text"][:200])
    else:
        print("CAPTION:", t["caption"])