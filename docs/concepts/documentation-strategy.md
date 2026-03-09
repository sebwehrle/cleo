# Documentation Strategy

## Current stack

This project uses MkDocs + mkdocstrings for documentation site generation.

## Why this shape

- Fast iteration with Markdown pages
- API reference generated from code/docstrings
- reST/Sphinx field-list docstrings (`:param:`, `:returns:`, `:raises:`) are parsed with `docstring_style: sphinx`
- Explicit knob/method tables for behavioral clarity where docstrings alone are insufficient
- CI-enforced docs build (`mkdocs build --strict`)

## Future migration path to Sphinx (Option 3)

To keep migration low-friction:

- Keep pages in portable Markdown structure (guides/concepts/reference)
- Avoid MkDocs-only macros/features in source content
- Treat docstrings as canonical API documentation source
- Keep runnable examples independent of site generator

When moving to Sphinx/MyST later, content should mostly transfer by remapping navigation/build configuration.
