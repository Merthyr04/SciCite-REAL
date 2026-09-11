"""Detect overlapping characters in the compiled PDF and extract references."""
import pdfplumber

pdf_path = r"c:\Users\Chen\.trae-cn\attachments\6aa27c5b9a72b590765f4e7a\fe225801-73a2-4ea3-b66a-f6d657357a8d_b185d159-22ba-4737-bed8-12a542ddfb78_SciCite_REAL.pdf"

with pdfplumber.open(pdf_path) as pdf:
    print("pages:", len(pdf.pages))
    # Page size vs column layout
    p0 = pdf.pages[0]
    print(f"page size: {p0.width:.0f} x {p0.height:.0f} pt")

    # Detect char bbox overlaps within each page
    for pno, page in enumerate(pdf.pages, 1):
        chars = page.chars
        # bucket by rounded top to limit O(n^2)
        n_overlap = 0
        samples = []
        for i, c in enumerate(chars):
            x0, y0, x1, y1 = c["x0"], c["top"], c["x1"], c["bottom"]
            for d in chars[i + 1:]:
                # vertical overlap > 60% of height and horizontal overlap > 2pt
                vy = min(y1, d["bottom"]) - max(y0, d["top"])
                hx = min(x1, d["x1"]) - max(x0, d["x0"])
                ch = (y1 - y0)
                if ch > 0 and vy > 0.6 * ch and hx > 2.0:
                    n_overlap += 1
                    if len(samples) < 12:
                        samples.append((c["text"], d["text"], round(x0), round(y0), c.get("fontname", "?")[:20]))
        if n_overlap:
            print(f"\npage {pno}: {n_overlap} overlapping char pairs")
            for s in samples:
                print("   ", s)

    # Column bounds on page 1: find gaps
    print("\n=== references (last page) ===")
    last = pdf.pages[-1]
    txt = last.extract_text() or ""
    print(txt[:4000])
