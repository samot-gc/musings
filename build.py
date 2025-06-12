import os
import urllib.parse
import yaml
from jinja2 import Environment, FileSystemLoader

PAPERS_DIR = "papers"
TEMPLATE_FILE = "index_template.html"
OUTPUT_HTML = "index.html"
OUTPUT_MD = "README.md"

def extract_yaml_front_matter(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines or lines[0].strip() != "---":
        return None

    yaml_lines = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        yaml_lines.append(line)

    try:
        return yaml.safe_load("".join(yaml_lines))
    except yaml.YAMLError:
        return None

def load_papers():
    papers = []
    for filename in os.listdir(PAPERS_DIR):
        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(PAPERS_DIR, filename)
        meta = extract_yaml_front_matter(filepath)

        if not meta:
            continue

        # Prepare authors string
        authors = meta.get("authors", "N/A")
        if isinstance(authors, list):
            authors = ", ".join(authors)

        # Prepare tags list
        tags = meta.get("tags", [])
        if tags is None:
            tags = []

        # Filename for link: URL-encoded .html version of markdown filename
        html_filename = urllib.parse.quote(filename.replace(".md", ".html"))

        paper_info = {
            "title": meta.get("parent", os.path.splitext(filename)[0]),
            "authors": authors,
            "year": meta.get("year"),  # keep None if missing for sorting
            "tags": tags,
            "filename": html_filename,
        }

        papers.append(paper_info)

    return papers

def build_html(papers):
    # Sort by year desc (None treated as 0), then title asc
    def sort_key(p):
        year = p["year"] if isinstance(p["year"], int) else 0
        return (-year, p["title"].lower())

    papers_sorted = sorted(papers, key=sort_key)

    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template(TEMPLATE_FILE)

    all_tags = sorted({tag for paper in papers_sorted for tag in paper["tags"]})

    output_html = template.render(papers=papers_sorted, all_tags=all_tags)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(output_html)

def build_markdown(papers):
    # Sort by year desc then title asc
    def sort_key(p):
        year = p["year"] if isinstance(p["year"], int) else 0
        return (-year, p["title"].lower())

    papers_sorted = sorted(papers, key=sort_key)

    lines = []
    lines.append("# Musings\n")
    lines.append("For a searchable, filterable, and sortable index, please visit [Musings Index](https://samot-gc.github.io/musings/).\n")
    lines.append("\n")
    lines.append("| Title | Year | Authors | Tags |")
    lines.append("|-------|------|---------|------|")

    def md_escape(text):
        return str(text).replace('|', '\\|')

    for paper in papers_sorted:
        title = paper["title"] or "N/A"
        filename = paper["filename"]
        year = paper["year"] if paper["year"] is not None else "N/A"
        authors = paper["authors"] or "N/A"
        tags = paper["tags"] or []
        tags_str = ", ".join(tags) if tags else "N/A"

        lines.append(
            f"| [{md_escape(title)}](papers/{md_escape(filename)}) | {md_escape(year)} | {md_escape(authors)} | {md_escape(tags_str)} |"
        )

    readme_content = "\n".join(lines)

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(readme_content)

def main():
    papers = load_papers()
    build_html(papers)
    build_markdown(papers)

if __name__ == "__main__":
    main()
