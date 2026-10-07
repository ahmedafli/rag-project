import sqlite3

con = sqlite3.connect("documents.db")
con.row_factory = sqlite3.Row

rows = con.execute("SELECT * FROM documents").fetchall()
cols = rows[0].keys() if rows else []

if not rows:
    print("Table 'documents' is empty.")
else:
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))

    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))

con.close()
