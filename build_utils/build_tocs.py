from bs4 import BeautifulSoup
import os
import re

PAPERS_DIR = "papers"

def insert_toc_in_file(folder, filename, print_filename=False):
    html_path = os.path.join(folder, filename + ".html")
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # Remove ToC if it already exists
    toc_pattern = re.compile(
        r'<h2>Table of Contents</h2>\s*<nav id="toc">.*?</nav>',
        re.DOTALL | re.IGNORECASE
    )
    html_cleaned = toc_pattern.sub('', html)

    soup = BeautifulSoup(html_cleaned, 'html.parser')

    # Find all h2 headers
    h2_headers = soup.find_all('h2')
    if not h2_headers:
        print(f"No <h2> headers found in {html_path}, skipping ToC insertion.")
        return

    # Build ToC nav element
    toc_nav = soup.new_tag("nav", id="toc")
    ul = soup.new_tag("ul")
    toc_nav.append(ul)

    def ensure_id(tag):
        if not tag.has_attr('id'):
            safe_id = tag.get_text().strip().lower().replace(' ', '-').replace('.', '')
            tag['id'] = safe_id
        return tag['id']

    for h2 in h2_headers:
        anchor_id = ensure_id(h2)
        li = soup.new_tag("li")
        a = soup.new_tag("a", href=f"#{anchor_id}")
        a.string = h2.get_text()
        a['style'] = "text-decoration: underline; color: inherit;"
        li.append(a)
        ul.append(li)

    # Insert ToC heading and ToC nav immediately before first h2
    first_h2 = h2_headers[0]
    toc_heading = soup.new_tag('h2')
    toc_heading.string = "Table of Contents"
    first_h2.insert_before(toc_heading)
    first_h2.insert_before(toc_nav)

    # Add minimal CSS for ToC list indent matching normal bullets
    style_tag = soup.new_tag('style')
    style_tag.string = '''
#toc ul {
    list-style-type: disc;
}
#toc a:hover {
    text-decoration: none;
}
'''
    if soup.head:
        soup.head.append(style_tag)
    else:
        head = soup.new_tag('head')
        head.append(style_tag)
        soup.insert(0, head)

    # Overwrite file with ToC inserted
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))

    if print_filename:
        print(f"Inserted ToC in {filename}")

def build_tocs(papers):
    for paper in papers:
        # html_file = os.path.join(PAPERS_DIR, paper["filename_raw"] + ".html")
        insert_toc_in_file(PAPERS_DIR, paper["filename_raw"])