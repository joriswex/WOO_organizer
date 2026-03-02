from pdf_import_reader import load_pdf

docs = load_pdf()

print()
for code, doc in docs.items():
    redactions = doc["redaction_codes"]
    print(f"=== Document {code} — {doc['category']} — {len(doc['pages'])} page(s): {doc['pages']} ===")
    if redactions:
        print(f"    Redaction codes: {redactions}")
    print(doc["text"][:400])
    if len(doc["text"]) > 400:
        print("    [...]")
    print()
