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
    
    # Sort by year desc (None treated as 0), then title asc
    def sort_key(p):
        year = p["year"] if isinstance(p["year"], int) else 0
        return (-year, p["title"].lower())
    
    papers_sorted = sorted(papers, key=sort_key)
    
    return papers_sorted

def build_html(papers):
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template(TEMPLATE_FILE)

    all_tags = sorted({tag for paper in papers for tag in paper["tags"]})

    output_html = template.render(papers=papers, all_tags=all_tags)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(output_html)

def build_markdown(papers, output_file="README.md"):
    readme_lines = []
    readme_lines.append("**Note**: A searchable, filterable, and sortable version of this index is available [here](./index.html).\n")
    readme_lines.append("\n")
    readme_lines.append("| Title | Year | Authors | Tags |")
    readme_lines.append("|-------|------|---------|------|")

    for paper in papers:
        title = paper["title"]
        year = paper["year"]
        authors = paper["authors"]
        tags = ", ".join(paper["tags"])
        md_filename = urllib.parse.quote(paper["filename"].replace(".html", ".md"))
        github_url = f"https://github.com/samot-gc/musings/blob/main/papers/{md_filename}"
        title_link = f"[{title}]({github_url})"

        readme_lines.append(f"| {title_link} | {year} | {authors} | {tags} |")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))


def main():
    papers = load_papers()
    build_html(papers)
    build_markdown(papers)

if __name__ == "__main__":
    main()
