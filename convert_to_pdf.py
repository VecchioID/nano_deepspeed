"""
Convert all markdown documents in the project to PDF.
Usage: python convert_to_pdf.py
"""

import os, re, glob
from pathlib import Path

import markdown
from weasyprint import HTML, CSS


base_dir = Path(__file__).parent
output_dir = base_dir / "pdf_output"
output_dir.mkdir(exist_ok=True)

md_files = sorted(base_dir.glob("0*-*.md"))

html_template = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
@page {{
  size: A4;
  margin: 2.2cm 2.5cm 2.2cm 2.5cm;
  @bottom-center {{
    content: counter(page) " / " counter(pages);
    font-size: 10px;
    color: #999;
    font-family: "Noto Sans CJK SC";
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: "Noto Sans CJK SC", "Noto Sans", sans-serif;
  font-size: 12pt;
  line-height: 1.7;
  color: #1a1a1a;
  counter-reset: h2-counter;
}}
h1 {{
  font-size: 22pt;
  text-align: center;
  margin-bottom: 1.5em;
  padding-bottom: 0.5em;
  border-bottom: 3px solid #2563eb;
  color: #1e3a8a;
  page-break-before: always;
}}
h1:first-of-type {{ page-break-before: avoid; }}
h2 {{
  font-size: 16pt;
  margin-top: 1.8em;
  margin-bottom: 0.6em;
  padding-bottom: 0.3em;
  border-bottom: 1px solid #cbd5e1;
  color: #1e40af;
}}
h3 {{
  font-size: 13pt;
  margin-top: 1.2em;
  margin-bottom: 0.4em;
  color: #334155;
}}
h4 {{
  font-size: 12pt;
  margin-top: 0.8em;
  color: #475569;
}}
p {{ margin: 0.5em 0; text-align: justify; }}
ul, ol {{ margin: 0.4em 0; padding-left: 1.8em; }}
li {{ margin: 0.2em 0; }}
code {{
  font-family: "Source Code Pro", "Cascadia Code", "JetBrains Mono", "Consolas", monospace;
  font-size: 10pt;
  background: #f1f5f9;
  padding: 0.15em 0.4em;
  border-radius: 3px;
}}
pre {{
  background: #0f172a;
  color: #e2e8f0;
  padding: 0.8em 1em;
  border-radius: 6px;
  font-size: 9.5pt;
  line-height: 1.5;
  overflow-x: auto;
  page-break-inside: avoid;
}}
pre code {{ background: none; padding: 0; font-size: inherit; color: inherit; }}
blockquote {{
  border-left: 4px solid #2563eb;
  margin: 0.8em 0;
  padding: 0.5em 1em;
  background: #f0f7ff;
  color: #1e40af;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 0.8em 0;
  font-size: 10.5pt;
  page-break-inside: avoid;
}}
th, td {{
  border: 1px solid #cbd5e1;
  padding: 0.4em 0.6em;
  text-align: left;
}}
th {{
  background: #2563eb;
  color: white;
  font-weight: 600;
}}
tr:nth-child(even) {{ background: #f8fafc; }}
hr {{
  border: none;
  border-top: 2px dashed #cbd5e1;
  margin: 1em 0;
}}
img {{ max-width: 100%; }}
</style>
</head>
<body>
{content}
</body>
</html>
"""

extensions = [
    "markdown.extensions.extra",
    "markdown.extensions.codehilite",
    "markdown.extensions.tables",
    "markdown.extensions.fenced_code",
    "markdown.extensions.toc",
    "markdown.extensions.nl2br",
]


def md_to_pdf(md_path, pdf_path):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(md_text, extensions=extensions)

    full_html = html_template.format(content=html_body)

    HTML(string=full_html).write_pdf(str(pdf_path))
    return pdf_path


def main():
    total = len(md_files)
    for i, md_file in enumerate(md_files, 1):
        pdf_name = md_file.stem + ".pdf"
        pdf_path = output_dir / pdf_name
        print(f"[{i}/{total}] Converting: {md_file.name} -> {pdf_name}")
        md_to_pdf(md_file, pdf_path)
        print(f"    -> Done: {pdf_path}")

    print(f"\nAll converted! PDFs saved to: {output_dir}/")
    print("Files:")
    for p in sorted(output_dir.glob("*.pdf")):
        mb = p.stat().st_size / 1e6
        print(f"  {p.name} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
