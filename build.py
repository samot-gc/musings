from build_utils.build_index import build_html, build_markdown
from build_utils.build_tocs import build_tocs
from build_utils.build_paperlist import build_paperlist

def main():
    papers = build_paperlist()
    
    # print("Building ToCs...")
    # build_tocs(papers)
    
    print("Building HTML index...")
    build_html(papers)

    print("Building Markdown index...")
    build_markdown(papers)

if __name__ == "__main__":
    main()
