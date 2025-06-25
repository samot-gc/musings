import os
import yaml
import urllib.parse

PAPERS_DIR = "papers"

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

def build_paperlist():
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
        
        filename_raw = os.path.splitext(filename)[0]
        paper_info = {
            "title": meta.get("parent", filename_raw),
            "authors": authors,
            "year": meta.get("year"),  # keep None if missing for sorting
            "method": meta.get("method", "n/a"),
            "tags": tags,
            "filename_raw": filename_raw,  # raw .md filename
            "filename_url": urllib.parse.quote(filename_raw),
        }

        papers.append(paper_info)

    # Sort by year desc (None treated as 0), then title asc
    def sort_key(p):
        year = p["year"] if isinstance(p["year"], int) else 0
        return (-year, p["title"].lower())

    papers_sorted = sorted(papers, key=sort_key)

    return papers_sorted