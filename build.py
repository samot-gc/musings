import os
import urllib.parse
import yaml
from jinja2 import Environment, FileSystemLoader

PAPERS_DIR = "papers"
TEMPLATE_FILE = "index_template.html"
OUTPUT_FILE = "index.html"

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

def main():
    papers = []

    for filename in os.listdir(PAPERS_DIR):
        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(PAPERS_DIR, filename)
        meta = extract_yaml_front_matter(filepath)

        if not meta:
            continue

        paper_info = {
            "title": meta.get("parent", os.path.splitext(filename)[0]),
            "authors": meta.get("authors", "N/A"),
            "year": meta.get("year", "N/A"),
            "tags": meta.get("tags", []),
            "filename": urllib.parse.quote(filename.replace(".md", ".html"))
        }

        papers.append(paper_info)

    # Sort papers by year (descending) then title
    papers.sort(key=lambda p: (str(p["year"]), p["title"]), reverse=True)

    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template(TEMPLATE_FILE)

    all_tags = sorted({tag for paper in papers for tag in paper["tags"]})
    output_html = template.render(papers=papers, all_tags=all_tags)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output_html)

if __name__ == "__main__":
    main()
