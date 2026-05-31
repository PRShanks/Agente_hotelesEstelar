# CLAUDE.md — Hoteles Estelar Chat Assistant

## Project Overvieww

Enterprise chatbot for Hoteles Estelar S.A. that provides Q&A, executive summaries, and FAQ generation based on financial and corporate reports. Built incrementally as part of a university course on Generative AI techniques.

**Stack:** Python 3.12 · uv · LangChain · Anthropic Claude + Ollama · Supabase pgvector · Streamlit

## Key Architecture Decisions

- **Factory pattern** for LLM creation (`llm/factory.py`): supports switching between Claude and Ollama without changing business logic.
- **Dependency injection**: all domain functions accept `llm: BaseChatModel | None` — never instantiate LLMs inside business logic.
- **Prompts separated**: all system prompts and templates live in `llm/prompts.py` and `system_prompt.txt`. Never hardcode prompts in chain logic.
- **External system prompt**: `system_prompt.txt` is loaded at runtime so non-technical users can edit behavior without touching code.
- **Semantic retrieval via Supabase pgvector**: uses `SupabaseVectorStore` with OpenAI/Ollama embeddings for semantic search. Vector dimensions match the model: 768 for nomic-embed-text, 1536 (or configured) for text-embedding-3-small.
- **Tool-calling router**: `bind_tools([query_financiero])` lets the LLM decide between RAG and deterministic SQLite financial data. Structured output with `method="json_mode"` + retry on validation error.

## Conventions

### Code Style

- **Docstrings are MANDATORY** on every module, class, and public function. Use the triple-quote format with description, parameters, and return types.
- **High cohesion, low coupling**: each module does ONE thing. When adding a feature, create a new focused module rather than extending an existing one.
- **Type hints** on all public function signatures. Use `str | None` not `Optional[str]`, `list[str]` not `List[str]`.
- **Import order**: stdlib → third-party → local, separated by blank lines.

### Package Management

- Use **uv exclusively**. Never run `pip install` directly.
- After adding any dependency, run `uv lock` to update the lockfile.
- Keep `pyproject.toml` name, version, and description updated.

### File Organization

```text
llm/          → Domain logic (LLM chains, data loading, prompts)
app/          → Presentation layer (Streamlit dashboard only)
scripts/      → Data pipeline scripts (scraping, consolidation)
data/         → Data files (raw and processed, gitignored where needed)
tests/        → Test files mirroring llm/ structure
```

- Never import from `app/` in `llm/`. The dependency direction is `app/` → `llm/` → `data/`.
- Business logic goes in `llm/`. UI orchestration goes in `app/`.

### Message Flow

The project uses direct message lists instead of chains for the main Q&A flow:

```python
mensajes = [
    ("system", system_prompt),
    *historial,
    ("human", f"...{contexto}...{pregunta}"),
]
resultado = _invoke_estructurado(llm, mensajes)
```

- For tool-calling, `bind_tools([tool])` is used to expose tools to the LLM.
- Structured output uses `with_structured_output(model, method="json_mode")` with retry on ValidationError.
- Chains (`|` operator) are still used in summarizer and FAQ generator via `StrOutputParser()`.

## Common Tasks

### Adding a new LLM capability

1. Create `llm/new_capability.py` with a single exported function.
2. Add the prompt to `llm/prompts.py`.
3. Accept `llm: BaseChatModel | None = None` parameter, default via `crear_llm()`.
4. Add tests in `tests/test_new_capability.py`.
5. Wire into `app/dashboard.py` if it needs a UI.
6. Update `Makefile` if there's a new run command.

### Adding a new LLM provider

1. Add the provider name and models to `llm/factory.py`.
2. Add the new `langchain-<provider>` dependency to `pyproject.toml`.
3. Run `uv lock`.

### Running the project

```bash
make dev          # Start Streamlit dashboard
make run-qa       # Run Q&A chain standalone
make run-summary  # Run summarizer standalone
make run-faq      # Run FAQ generator standalone
make lint         # Run linters
make test         # Run tests
make setup        # Initial setup (venv + dependencies)
```

## Important Notes

- Python 3.12 is REQUIRED. Dependency `faiss-cpu` (future use) does not support 3.13+.
- `.env` contiene las credenciales (ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY) — nunca commitearlo.
- El tool-calling loop usa `bind_tools([query_financiero])` + `with_structured_output(method="json_mode")`. Si el LLM omite campos, el retry automático le informa cuáles faltan y lo reintenta.
- `data/processed/estelar_consolidado.txt` es auto-generado desde `data/estelar_reportes/*.md` — nunca editarlo manualmente.
- `data/processed/metricas_financieras.db` es auto-generado desde `data/estelar_reportes/metricas_financieras.csv` — nunca editarlo manualmente.
- Cuando se consulte documentación de LangChain, siempre usar la versión más reciente. La API cambia frecuentemente entre versiones mayores.
