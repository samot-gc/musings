# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic paper summaries repository with a static site generator. It maintains markdown summaries of AI/ML research papers (focused on LLM reasoning, RL, and latent reasoning) and generates:
- An interactive HTML index page with search and tag filtering
- A GitHub-friendly README.md with a table of papers

The site is published at: https://samot-gc.github.io/summaries-with-sam/

## Development Setup

**Dependencies:** jinja2, PyYAML, beautifulsoup4

### Option 1: Install Globally

**Using pip:**
```bash
pip install jinja2 pyyaml beautifulsoup4
# Note: May require --break-system-packages on some systems (WSL, modern Ubuntu/Debian)
```

**Using apt (Debian/Ubuntu/WSL):**
```bash
sudo apt install python3-jinja2 python3-yaml python3-bs4
```

### Option 2: Use Virtual Environment

**Standard venv:**
```bash
# Create virtual environment (if doesn't exist)
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac/WSL
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install jinja2 pyyaml beautifulsoup4
```

**Using uv (faster alternative):**
```bash
# Create and activate venv
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install jinja2 pyyaml beautifulsoup4
```

### Build Command

```bash
# Generate index.html and README.md from paper metadata
python build.py
```

## Architecture

### Paper Format

Papers are markdown files in `papers/` with YAML front matter:

```yaml
---
title: "Full Paper Title"
lab: "Lab/Organization name"  # Can be a list or string
date: 202510  # YYYYMM format (e.g., 202510 for October 2025)
method: "TRM"  # Short method abbreviation
tags:
  - hierarchy
  - latent reasoning
  - reasoning
---

Paper summary content in markdown...
```

**Required YAML fields:** title, lab, date, method, tags

### Build System Pipeline

The build system is a three-stage pipeline:

1. **`build_utils/build_paperlist.py`**
   - Scans `papers/` directory for `.md` files
   - Extracts YAML front matter using PyYAML
   - Creates paper metadata objects with URL-encoded filenames
   - Sorts by date (descending), then title (ascending)

2. **`build_utils/build_index.py`**
   - **`build_html(papers)`**: Renders interactive HTML index using Jinja2
     - Search box filters by title
     - Tag buttons for filtering
     - Sortable date column
     - All client-side JavaScript (no server needed)
   - **`build_markdown(papers)`**: Generates simple markdown table for README
     - Links to GitHub blob URLs

3. **`build_utils/index_template.html`**
   - Jinja2 template with embedded CSS and JavaScript
   - Vanilla JS for interactivity (no frameworks)

### Generated Files

**IMPORTANT:** These files are build artifacts and should NEVER be manually edited:
- `index.html` - Generated from template
- `README.md` - Generated from paper metadata

Always regenerate by running `python build.py` after changes.

## Adding New Papers

1. Create a new `.md` file in `papers/` with the required YAML front matter
2. Write the paper summary in markdown below the YAML section
3. Run `python build.py` to regenerate index files
4. Commit changes (including generated `index.html` and `README.md`)
5. Push to GitHub (triggers automatic GitHub Pages deployment)

## Important Conventions

**File Naming:**
- Spaces are allowed in paper filenames (e.g., `Rethinking Thinking.md`)
- The build system handles URL encoding automatically
- HTML files mirror markdown filenames (e.g., `HRM.md` → `HRM.html`)

**Date Format:**
- Use `YYYYMM` format (e.g., `202510` for October 2025, `202501` for January 2025)
- This enables proper chronological sorting

**Lab Names:**
- Common values: Meta, Samsung, DeepSeek-AI, mixed, "OT" (own)
- Can be a list for multi-lab collaborations: `["Meta", "Stanford"]`

**Method Abbreviations:**
- Use short, memorable acronyms (e.g., TRM, HRM, COCONUT, GRPO, PPO)

**Tags:**
- Common tags: `reasoning`, `RL`, `training`, `latent reasoning`, `hierarchy`, `math`
- Tags enable filtering in the interactive HTML index

## Project Structure Notes

**`potm/` Directory:**
- Contains "Paper of the Month" drafts with a different workflow
- Uses a separate template (`paper_summary_template.md`)
- Different YAML schema (includes `paper_authors`, `orgs`, `paper_link`, etc.)
- Do not modify unless working on POTM content

**`build_utils/build_tocs.py`:**
- Table of contents generation module (currently disabled in main build)
- Can generate ToC `<nav>` elements from `<h2>` headers in HTML files
- Not currently used in the standard workflow

## GitHub Pages Deployment

- Repository: https://github.com/samot-gc/summaries-with-sam.git
- Site URL: https://samot-gc.github.io/summaries-with-sam/
- Uses default GitHub Pages (no Jekyll)
- `index.html` is served as the homepage
- Deployment triggers automatically on push to main branch

## Auxiliary Files and Claude Code

**Location for auxiliary files:** `.claude/`

When creating auxiliary files (notes, analysis, temp files, etc.), store them in the `.claude/` directory:
- Documentation files (like this CLAUDE.md) are committed to git
- Settings files (`.claude/settings.local.json`, `.claude/*.local.*`) are git-ignored
- This keeps auxiliary files organized and separate from repository content

## Technical Stack

- **Python 3.12+** for build scripts
- **Jinja2** for HTML templating
- **PyYAML** for front matter parsing
- **BeautifulSoup4** for HTML manipulation (ToC generation)
- **Vanilla HTML/CSS/JavaScript** for the static site (no frameworks)
- **Markdown** for paper content

## Git Configuration

This repository is configured to use SSH for GitHub operations:
- Remote URL: `git@github.com:samot-gc/summaries-with-sam.git`
- GitHub CLI (`gh`) is used for authentication
- Main branch is `main`

If you need to clone the repository:
```bash
git clone git@github.com:samot-gc/summaries-with-sam.git
```
