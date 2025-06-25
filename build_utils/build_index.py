import os
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = "build_utils"
TEMPLATE_FILE = "index_template.html"
OUTPUT_HTML = "index.html"
OUTPUT_MD = "README.md"

def build_html(papers):
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(TEMPLATE_FILE)

    all_tags = sorted({tag for paper in papers for tag in paper["tags"]})

    output_html = template.render(papers=papers, all_tags=all_tags)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(output_html)

def build_markdown(papers, output_file="README.md"):
    readme_lines = []
    readme_lines.append("A searchable, filterable, and sortable version of this index is available on the [GitHub page](https://samot-gc.github.io/musings/index.html).\n")
    readme_lines.append("\n")
    readme_lines.append("| Title | Year | Authors | Tags |")
    readme_lines.append("|-------|------|---------|------|")

    for paper in papers:
        title = paper["title"]
        year = paper["year"]
        authors = paper["authors"]
        tags = ", ".join(paper["tags"])
        md_filename = paper["filename_url"] + ".md"
        github_url = f"https://github.com/samot-gc/musings/blob/main/papers/{md_filename}"
        title_link = f"[{title}]({github_url})"

        readme_lines.append(f"| {title_link} | {year} | {authors} | {tags} |")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))
