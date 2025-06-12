import os
import frontmatter
from jinja2 import Template
from pathlib import Path
import urllib.parse

# Configuration
NOTES_DIR = "papers"
OUTPUT_FILE = "index.html"
REPO_URL_BASE = "https://github.com/samot-gc/musings/blob/main/"

# Load all markdown notes
papers = []
for file in Path(NOTES_DIR).rglob("*.md"):
    post = frontmatter.load(file)
    relative_path = str(file).replace("\\", "/")  # cross-platform
    encoded_path = urllib.parse.quote(relative_path)
    github_url = REPO_URL_BASE + encoded_path
    papers.append({
        "title": post.get("parent", file.stem),  # Read title from 'parent' field
        "authors": ", ".join(post.get("authors", [])),
        "tags": post.get("collections", []),  # Using 'collections' instead of 'tags'
        "year": post.get("year", ""),
        "venue": post.get("venue", ""),
        "path": github_url,
    })

# HTML Template using List.js for sorting/filtering/search
html_template = Template("""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <title>Paper Notes Index</title>
    <script src=\"https://cdnjs.cloudflare.com/ajax/libs/list.js/2.3.1/list.min.js\"></script>
    <style>
        body { font-family: sans-serif; padding: 2rem; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 0.5rem; border: 1px solid #ccc; text-align: left; }
        input { margin-bottom: 1rem; padding: 0.5rem; width: 300px; }
    </style>
</head>
<body>
    <h1>ML Paper Notes</h1>
    <input class=\"search\" placeholder=\"Search papers...\" />
    <div id=\"papers\">
        <table>
            <thead>
                <tr>
                    <th><button class=\"sort\" data-sort=\"title\">Title</button></th>
                    <th><button class=\"sort\" data-sort=\"authors\">Authors</button></th>
                    <th><button class=\"sort\" data-sort=\"tags\">Tags</button></th>
                    <th><button class=\"sort\" data-sort=\"year\">Year</button></th>
                    <th><button class=\"sort\" data-sort=\"venue\">Venue</button></th>
                </tr>
            </thead>
            <tbody class=\"list\">
                {% for paper in papers %}
                <tr>
                    <td class=\"title\"><a href=\"{{ paper.path }}\" target=\"_blank\">{{ paper.title }}</a></td>
                    <td class=\"authors\">{{ paper.authors }}</td>
                    <td class=\"tags\">{{ paper.tags | join(", ") }}</td>
                    <td class=\"year\">{{ paper.year }}</td>
                    <td class=\"venue\">{{ paper.venue }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    <script>
        var options = {
            valueNames: [ 'title', 'authors', 'tags', 'year', 'venue' ]
        };
        var paperList = new List('papers', options);
    </script>
</body>
</html>
""")

# Write HTML index
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(html_template.render(papers=sorted(papers, key=lambda x: x['year'], reverse=True)))

print(f"Generated {OUTPUT_FILE} with {len(papers)} papers.")
