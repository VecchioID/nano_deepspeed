"""
Convert all markdown learning documents to professional PDFs.
Uses markdown → HTML → WeasyPrint pipeline with proper typography.
"""

import os, re, glob
from pathlib import Path
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

base_dir = Path(__file__).parent
output_dir = base_dir / "pdf_output"
output_dir.mkdir(exist_ok=True)

# Files to convert (in order)
md_files = [
    base_dir / "01-LLM训练挑战与并行策略概览.md",
    base_dir / "02-DeepSpeed-ZeRO优化详解.md",
    base_dir / "03-DeepSpeed高级功能与实战.md",
    base_dir / "04-框架对比与选型指南.md",
    base_dir / "LAB-动手实验.md",
    base_dir / "zero_from_scratch/LAB-从零实现ZeRO.md",
]

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
@page {{
  size: A4;
  margin: 2cm 2.2cm 2cm 2.2cm;
  @bottom-center {{
    content: counter(page);
    font-size: 9pt;
    color: #888;
    font-family: "Noto Sans CJK SC", sans-serif;
  }}
}}

* {{ box-sizing: border-box; }}

body {{
  font-family: "Noto Sans CJK SC", "Source Han Sans SC", sans-serif;
  font-size: 11pt;
  line-height: 1.75;
  color: #1a1a1a;
}}

h1 {{
  font-size: 20pt;
  text-align: center;
  margin: 2em 0 1em 0;
  padding-bottom: 0.4em;
  border-bottom: 3px solid #2563eb;
  color: #1e3a8a;
  page-break-before: always;
  page-break-after: avoid;
}}
h1:first-of-type {{ page-break-before: avoid; }}

h2 {{
  font-size: 15pt;
  margin: 1.5em 0 0.5em 0;
  padding-bottom: 0.3em;
  border-bottom: 1px solid #cbd5e1;
  color: #1e40af;
  page-break-after: avoid;
}}

h3 {{
  font-size: 13pt;
  margin: 1.2em 0 0.4em 0;
  color: #334155;
  page-break-after: avoid;
}}

h4 {{
  font-size: 11.5pt;
  margin: 0.8em 0 0.3em 0;
  color: #475569;
}}

p {{ margin: 0.4em 0; text-align: justify; }}

ul, ol {{ margin: 0.3em 0; padding-left: 1.6em; }}
li {{ margin: 0.15em 0; }}

code {{
  font-family: "Source Code Pro", "Cascadia Code", "Consolas", monospace;
  font-size: 9pt;
  background: #f1f5f9;
  padding: 0.1em 0.35em;
  border-radius: 2px;
  word-break: break-all;
}}

pre {{
  background: #0f172a;
  color: #e2e8f0;
  padding: 0.6em 0.8em;
  border-radius: 4px;
  font-size: 8.5pt;
  line-height: 1.45;
  overflow-x: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
  page-break-inside: avoid;
  margin: 0.5em 0;
}}

pre code {{
  background: none;
  padding: 0;
  font-size: inherit;
  color: inherit;
}}

blockquote {{
  border-left: 4px solid #2563eb;
  margin: 0.6em 0;
  padding: 0.3em 0.8em;
  background: #f0f7ff;
  color: #1e40af;
  page-break-inside: avoid;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  margin: 0.6em 0;
  font-size: 9.5pt;
  page-break-inside: avoid;
  table-layout: fixed;
}}

th, td {{
  border: 1px solid #94a3b8;
  padding: 0.3em 0.4em;
  text-align: left;
  word-wrap: break-word;
  overflow-wrap: break-word;
}}

th {{
  background: #2563eb;
  color: white;
  font-weight: 600;
}}

tr:nth-child(even) {{ background: #f8fafc; }}

hr {{
  border: none;
  border-top: 1.5px dashed #94a3b8;
  margin: 0.8em 0;
}}

img {{
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.5em auto;
}}

strong {{ color: #1e293b; }}
em {{ color: #475569; }}

/* Inline code in tables */
td code, th code {{
  font-size: 8.5pt;
  word-break: break-all;
}}

/* Keep code blocks in tables compact */
td pre {{
  margin: 0;
  padding: 0.2em 0.4em;
  font-size: 8pt;
}}
</style>
</head>
<body>
{content}
</body>
</html>
"""

markdown_extensions = [
    "markdown.extensions.extra",
    "markdown.extensions.codehilite",
    "markdown.extensions.tables",
    "markdown.extensions.fenced_code",
    "markdown.extensions.toc",
    "markdown.extensions.nl2br",
    "markdown.extensions.sane_lists",
]


def preprocess_md(text):
    """Fix table formatting and other markdown issues before conversion."""
    lines = text.split('\n')
    result = []
    in_code_block = False
    in_table = False

    for i, line in enumerate(lines):
        # Track code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        # Trim trailing spaces in table rows
        if line.strip().startswith('|') and line.strip().endswith('|'):
            result.append(line)
            in_table = True
            continue
        else:
            if in_table and line.strip() == '':
                in_table = False

        result.append(line)

    return '\n'.join(result)


def md_to_pdf(md_path, pdf_path):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    md_text = preprocess_md(md_text)
    html_body = markdown.markdown(md_text, extensions=markdown_extensions)
    full_html = HTML_TEMPLATE.format(content=html_body)

    font_config = FontConfiguration()
    doc = HTML(string=full_html)
    doc.write_pdf(
        str(pdf_path),
        font_config=font_config,
        presentational_hints=True,
    )
    return pdf_path


def main():
    total = len(md_files)
    pdf_paths = []

    for i, md_file in enumerate(md_files, 1):
        pdf_name = md_file.stem + ".pdf"
        pdf_path = output_dir / pdf_name
        print(f"[{i}/{total}] {md_file.name} → {pdf_name}...")
        md_to_pdf(str(md_file), str(pdf_path))
        size_mb = pdf_path.stat().st_size / 1e6
        pdf_paths.append(pdf_path)
        print(f"       ✓ {size_mb:.1f} MB")

    print(f"\n全部转换完成! {total} 个 PDF 已保存到 {output_dir}/")

    # Merge all into one
    try:
        import fitz
        merged = fitz.open()
        for p in pdf_paths:
            doc = fitz.open(p)
            merged.insert_pdf(doc)
            doc.close()
        merged_path = base_dir / "DeepSpeed-LLM训练框架系统学习.pdf"
        merged.save(merged_path)
        merged.close()
        print(f"合并为一个 PDF: {merged_path} ({merged_path.stat().st_size/1e6:.1f} MB)")
    except ImportError:
        print("PyMuPDF 未安装, 跳过合并")

    print("\n单个 PDF 文件列表:")
    for p in sorted(output_dir.glob("*.pdf")):
        if "DeepSpeed" not in p.name:
            print(f"  {p.name} ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
