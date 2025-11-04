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

        # Flatten lab if list
        lab = meta.get("lab")
        if isinstance(lab, list):
            meta["lab"] = ", ".join(lab)
        
        filename_raw = os.path.splitext(filename)[0]
        paper_info = {
            "title": meta.get("title", filename_raw),
            "lab": meta.get("lab", "mixed"),
            "date": meta.get("date"),  # keep None if missing for sorting
            "method": meta.get("method", "n/a"),
            "tags": meta.get("tags", []),
            "filename_raw": filename_raw,  # raw .md filename
            "filename_url": urllib.parse.quote(filename_raw),
        }

        papers.append(paper_info)

    # Sort by date desc (None treated as 0), then title asc
    def sort_key(p):
        date = p["date"] if isinstance(p["date"], int) else 0
        return (-date, p["title"].lower())

    papers_sorted = sorted(papers, key=sort_key)

    return papers_sorted