# LLM Wiki — Schema

This is the operating manual for the LLM maintaining this wiki. Read it before doing anything.

## Philosophy

This is a **personal knowledge base** that compounds over time. Every source ingested and every good answer generated should enrich the wiki permanently. The LLM's job is to do the bookkeeping — summarizing, cross-referencing, filing, keeping everything consistent — so the human can focus on curating sources and asking good questions.

- **Raw sources are immutable.** Read them, never modify them.
- **The wiki is the LLM's responsibility.** Write it, maintain it, keep it healthy.
- **Good answers get filed.** Don't let insights disappear into chat history.

## Directory Structure

```
llm_wiki/
├── AGENTS.md              # This file — wiki schema and operating instructions
├── raw/                   # Immutable source documents (articles, papers, PDFs, notes)
├── wiki/                  # All LLM-generated and maintained content
│   ├── index.md           # Master catalog — every page listed with summary
│   ├── log.md             # Chronological log of all operations
│   ├── overview.md        # High-level synthesis of everything in the wiki
│   ├── health/            # Physical health, mental health, habits
│   ├── goals/             # Short and long-term goals, progress tracking
│   ├── learning/          # Topics being studied, courses, skills
│   ├── reading/           # Book notes, article notes, reading lists
│   ├── journal/           # Time-based reflections and entries
│   ├── people/            # Important people, relationships
│   └── ideas/             # Brainstorms, project ideas, things to explore
```

New categories can be added organically as needed. When adding a category, update this file and `index.md`.

## Page Conventions

### File naming
- Use lowercase kebab-case: `deep-learning-basics.md`, `2026-q2-goals.md`
- Files go in the appropriate category directory under `wiki/`

### Frontmatter (required on every page)
```yaml
---
title: "Page Title"
type: entity | concept | summary | source-note | query | journal | goal | comparison
tags: [tag1, tag2]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/article-name.md]
status: draft | stable | needs-review
---
```

- `type`: What kind of page this is (required)
- `tags`: Lowercase, use existing tags from index.md when possible
- `sources`: Link to raw source files this page draws from (can be empty)
- `status`: `draft` (incomplete/rough), `stable` (reviewed, accurate), `needs-review` (may be outdated or contradicted)

### Linking
- Use standard markdown links with relative paths: `[Sleep Quality](health/sleep-quality.md)`
- Link liberally — cross-references are the wiki's connective tissue
- When you link to a page that doesn't exist yet, create a stub for it later during lint

### Content style
- Write in clear, concise markdown
- Use headings for structure (start at `##` — the page title is `#`)
- Include a **Key Points** section at the top of summary/entity pages as a bullet list
- Cite sources inline with `[src](raw/file.md)` links
- Note contradictions explicitly: `> ⚠️ Contradiction: [Source A] claims X, but [Source B] claims Y`
- Use `> [!note]` callouts sparingly for important caveats

## Operations

### 1. Ingest (processing a new source)

When the user adds a file to `raw/` and asks you to process it:

1. **Read the source** thoroughly. For PDFs, use PyPDF2 or PyMuPDF to extract text. If the PDF is too large to read inline, extract sections progressively.
2. **Extract figures**: 
   - For arXiv papers: download the LaTeX source from `arxiv.org/src/{id}`, extract original vector figures from the `figures/` directory, convert to PNG at 2-3× resolution via PyMuPDF, save to `wiki/assets/`
   - For non-arxiv sources: render the figure page from PDF, crop to the figure area only (no headers/footers/surrounding text). Never use full-page renders.
   - Reference figures as `![caption](../assets/filename.png)` with `[src](raw/paper.pdf)` attribution
3. **Discuss key takeaways** with the user — what stood out, what's interesting, what connects to existing wiki content. Ask what they want to emphasize
4. **Read `wiki/index.md`** to understand what's already in the wiki
5. **Create a source-note page** summarizing the source in the appropriate category. Include:
   - Key points
   - Full summary
   - Connections to existing wiki pages (with links)
   - Any contradictions with existing knowledge
6. **Update related pages** — entity pages, concept pages, topic summaries. A single source might touch 5-15 pages. For each existing page mentioned:
   - Add new information
   - Note contradictions
   - Strengthen or challenge existing claims
   - Update `updated` date and add the source to the `sources` list
7. **Update `wiki/index.md`** — add entry for the new page, update entries for modified pages
8. **Update `wiki/overview.md`** if the source adds significant new knowledge
9. **Append to `wiki/log.md`** — log the ingest with consistent prefix format
10. **Tell the user which pages were touched** so they can browse them

### Math & Formula Conventions

- Use **LaTeX math** for all formulas:
  - Display equations: `$$ ... $$` with `\begin{aligned}` for multi-line
  - Inline math: `$ ... $`
  - Use proper notation: `\times`, `\cdot`, `\sum`, `\prod`, `\frac`, `\sqrt`, `\text`, `\mathbb{E}`, `\propto`, `\to`
- Algorithmic pseudocode may remain in code blocks
- Never leave duplicate formulas (plain text + LaTeX side by side)

### 2. Query (answering a question from the wiki)

1. **Read `wiki/index.md`** to find relevant pages
2. **Read the relevant pages** (use LSP, search, or direct file reads)
3. **Synthesize an answer** with inline citations to wiki pages and raw sources
4. **Ask the user:** "Should I file this answer into the wiki?" If yes:
   - Create a new `query`-type page (or update an existing one) with the answer
   - Update `wiki/index.md` and `wiki/log.md`

### 3. Lint (wiki health check)

Periodically, when asked (or proactively suggest it after ~5 ingests):

1. **Check for contradictions** — search for claims across pages that disagree. Flag with `⚠️ Contradiction` callouts and add `needs-review` status
2. **Check for stale pages** — pages with `needs-review` status or old `updated` dates that newer sources might supersede
3. **Find orphan pages** — pages with no inbound links from other wiki pages (except index.md). Either add links from relevant pages or flag as orphan
4. **Find missing pages** — concepts or entities mentioned multiple times across pages but lacking their own page. Create stubs (draft status)
5. **Check for content gaps** — topics the user seems interested in but with thin coverage. Suggest new sources to look for
6. **Report findings** to the user with a prioritized list of issues and suggested actions

### 4. Updates from conversation

When the user shares information, insights, or decisions in conversation:

- **Offer to file it** — "Want me to add that to the wiki?"
- If yes, create or update the appropriate page, following the same discipline as ingest

## index.md Format

A table-based catalog, organized by category. Every wiki page appears here.

```markdown
# Index

*Last updated: 2026-05-02*

## Health (2 pages)
| Page | Summary | Type | Tags | Updated |
|------|---------|------|------|---------|
| [Sleep Quality](health/sleep-quality.md) | Factors affecting sleep, tracking data, interventions | concept | sleep, habits | 2026-05-02 |
| [Exercise Routine](health/exercise-routine.md) | Current workout plan, progress log | entity | fitness, habits | 2026-05-01 |

## Goals (1 page)
| Page | Summary | Type | Tags | Updated |
|------|---------|------|------|---------|
| ...
```

- Sort pages alphabetically within each category
- Update the `*Last updated*` timestamp at the top whenever the index changes
- Count pages per category in section headers

## log.md Format

Append-only chronological log. Every ingest, query (if filed), lint, and significant update gets an entry.

```markdown
# Log

## [2026-05-02] ingest | Article: Why We Sleep (Chapter 3)
- Created: health/sleep-stages.md
- Updated: health/sleep-quality.md, health/routine.md
- 3 pages touched

## [2026-05-02] ingest | Podcast: Huberman Lab — Dopamine
- Created: health/dopamine-basics.md, learning/motivation-systems.md
- Updated: goals/2026-q2-goals.md
- Flagged contradiction in health/routine.md regarding morning protocols
- 5 pages touched

## [2026-05-02] query | Comparison: meditation vs exercise for focus
- Filed answer as: health/focus-strategies-comparison.md
```

- Prefix: `## [YYYY-MM-DD] operation | Description`
- Always list pages created and updated
- Note contradictions flagged
- Keep entries concise but informative

## Shortcuts & Efficiency

- **Before answering any question**, always read `wiki/index.md` first so you know what's available
- **When touching multiple pages**, batch your reads (parallel) and writes to be efficient
- **When in doubt**, ask the user. Don't guess what they want emphasized or where to file something
- **Be conservative about creating new categories** — try to fit pages into existing categories first
- **Suggest lint** after approximately every 5 ingest operations or when you notice inconsistencies
