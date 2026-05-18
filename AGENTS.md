# AGENTS.md — Lean Codex Config for Python Quant / VectorBT Pro

## MCP / Tooling Rules
- Prefer MCP tools for symbol search, references, renames, and targeted edits
- Use symbol-level inspection before reading entire files
- Read only the minimum code needed to complete task
- When changing one function/class, avoid loading unrelated modules
- Prefer targeted patches over full-file rewrites
- Use filesystem search only when MCP cannot resolve symbol paths
- Prefer inspecting local wrappers/helpers before exploring vectorbtpro internals; assume issue is local until evidence suggests otherwise

## File Rules
- Never edit .venv, dist, build, cache, generated folders
- Never inspect CSV/parquet data files unless explicitly requested
- Prefer editing existing modules over creating new files
- Preserve current project structure

## Code Style
- Prefer vectorized pandas/numpy operations
- Prefer vectorbtpro-native methods where possible
- Keep functions small and composable
- Avoid unnecessary abstractions
- Keep imports minimal

## Performance Rules
- Avoid loops when vectorization is practical
- Be careful with memory-heavy dataframe copies
- Reuse existing arrays/series where possible

## Output Rules
- Prefer minimal diffs
- Patch exact functions/classes where possible
- Do not rewrite full files unless explicitly required
- Change only necessary lines
- Preserve formatting/style of surrounding code
- If scope needs expansion, ask first

## Context Efficiency
- Do not open large files unless necessary
- Do not inspect files under data/ and outputs/log/ unless requested
- Do not inspect these folders unless requested: validation/ analysis/ tests/ research/misc/ 
- Do not inspect entire codebase unless requested. Ask which folders/section you should look at to understand the context.
- Summarize findings briefly before making broad edits
- If multiple files may be relevant, inspect smallest likely target first
- Avoid loading third-party library source code unless required to solve task

## When Unsure
- Ask for target file(s)
- Ask for expected behaviour
- Do not guess hidden architecture