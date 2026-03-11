"""Enrichment pipeline: add embeddings and summaries to .acatome bundles.

Steps 6–7 from spec:
  6a. Per-block summaries (LLM)
  6b. Per-section summaries (distilled from block summaries)
  6c. Per-paper summary (distilled from section summaries)
  7.  Embed original text + all summaries, keyed by profile name.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

log = logging.getLogger("acatome_extract.enrich")

from acatome_extract.bundle import read_bundle, write_bundle

# Block types that skip embeddings
_SKIP_EMBED_TYPES = {"section_header", "title", "author", "equation", "junk"}

_BLOCK_PROMPT_TEMPLATE = (
    "Terse telegram-style summary, one line. "
    "Always respond in English regardless of source language. "
    "No articles, no filler ('This passage', 'The authors'). "
    "Core claim first; detail after semicolon. "
    "Must make sense if truncated at semicolon."
)

_PAPER_PROMPT_TEMPLATE = (
    "3-5 telegram-style lines, broadest claim to finest detail. "
    "Always respond in English regardless of source language. "
    "No articles, no filler. Line 1 stands alone as summary. "
    "Each subsequent line narrows scope."
)


def enrich(
    bundle_path: str | Path,
    profiles: list[str] | None = None,
    summarize: bool = True,
    summarizer: str = "ollama/qwen3.5:9b",
) -> Path:
    """Enrich a .acatome bundle with embeddings and summaries.

    Args:
        bundle_path: Path to the .acatome bundle.
        profiles: Embedding profile names to compute (default: ["default"]).
        summarize: Whether to generate summaries.
        summarizer: litellm model spec ("ollama/qwen3.5:9b",
            "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-20250514", etc.).

    Returns:
        Path to the updated bundle.
    """
    bundle_path = Path(bundle_path)
    data = read_bundle(bundle_path)
    blocks = data.get("blocks", [])
    profiles = profiles or ["default"]

    from acatome_meta.config import load_config

    cfg = load_config()

    # Step 5½: Translate slug if title is non-Latin
    if summarize:
        bundle_path = _translate_slug(data, bundle_path, summarizer)

    # Step 6: Summarize
    if summarize:
        title = data.get("header", {}).get("title", "")
        summary_key = f"llm:{summarizer}"
        blocks = _summarize_blocks(blocks, summarizer, title=title, summary_key=summary_key)
        paper_summary = _summarize_paper(blocks, summarizer, summary_key=summary_key)
        data["enrichment_meta"] = data.get("enrichment_meta") or {}
        data["enrichment_meta"]["summarizer"] = summarizer
        # Store paper summaries as a dict keyed by method
        paper_summaries = data["enrichment_meta"].get("paper_summaries") or {}
        paper_summaries[summary_key] = paper_summary
        data["enrichment_meta"]["paper_summaries"] = paper_summaries
        # Backward compat: also write paper_summary as best pick
        from precis_summary import pick_best_summary
        data["enrichment_meta"]["paper_summary"] = pick_best_summary(paper_summaries)
        # Store prompt templates for provenance
        data["enrichment_meta"].setdefault("summary_prompts", {})[summary_key] = {
            "block": _BLOCK_PROMPT_TEMPLATE,
            "paper": _PAPER_PROMPT_TEMPLATE,
        }

    # Step 7: Embed
    for profile_name in profiles:
        profile_cfg = cfg.extract.profiles.get(profile_name)
        if not profile_cfg:
            continue
        embedder = _get_embedder(profile_cfg)
        if embedder is None:
            continue
        blocks = _embed_blocks(blocks, embedder, profile_name)

    data["blocks"] = blocks

    # Record enrichment metadata (including actual model names for sharing)
    if data.get("enrichment_meta") is None:
        data["enrichment_meta"] = {}
    profile_details = {}
    for pname in profiles:
        pcfg = cfg.extract.profiles.get(pname)
        if pcfg:
            profile_details[pname] = {"model": pcfg.model, "dim": pcfg.dim}
    data["enrichment_meta"]["profiles"] = profiles
    data["enrichment_meta"]["embedding_models"] = profile_details

    write_bundle(data, bundle_path)
    return bundle_path


# ---------------------------------------------------------------------------
# Non-Latin slug translation
# ---------------------------------------------------------------------------


def _is_non_latin(title: str) -> bool:
    """Return True if the title is predominantly non-Latin script."""
    if not title.strip():
        return False
    ascii_title = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    )
    ascii_letters = sum(1 for c in ascii_title if c.isalpha())
    total_letters = sum(1 for c in title if c.isalpha())
    if total_letters == 0:
        return False
    return ascii_letters / total_letters < 0.5


def _translate_slug(
    data: dict[str, Any], bundle_path: Path, summarizer: str
) -> Path:
    """If the title is non-Latin, ask the LLM for an English keyword and update the slug.

    Returns the (possibly renamed) bundle path.
    """
    header = data.get("header", {})
    title = header.get("title", "")
    if not _is_non_latin(title):
        return bundle_path

    llm = _get_llm(summarizer)
    if llm is None:
        log.warning("  [slug] no LLM available for slug translation")
        return bundle_path

    try:
        keyword = llm(
            "Translate this paper title to English, then pick the single most "
            "descriptive keyword (one lowercase word, no articles, no stopwords). "
            "Reply with ONLY that one word, nothing else.\n\n"
            f"Title: {title}"
        ).strip().lower()
        # Sanitize: keep only a-z
        keyword = re.sub(r"[^a-z]", "", keyword)
        if not keyword or len(keyword) < 2:
            log.warning("  [slug] LLM returned unusable keyword %r", keyword)
            return bundle_path
    except Exception as exc:
        log.warning("  [slug] LLM translation failed: %s", exc)
        return bundle_path

    from acatome_extract.ids import make_slug

    old_slug = header.get("slug", "")
    new_slug = make_slug(
        header.get("authors", []), header.get("year"), keyword
    )
    if new_slug == old_slug:
        return bundle_path

    log.info("  [slug] %s → %s (translated from %r)", old_slug, new_slug, title[:40])
    header["slug"] = new_slug
    data["header"] = header

    # Rename bundle file
    new_path = bundle_path.parent / f"{new_slug}.acatome"
    if not new_path.exists():
        bundle_path.rename(new_path)
        # Also rename companion PDF if present
        old_pdf = bundle_path.with_suffix(".pdf")
        if old_pdf.exists():
            new_pdf = new_path.with_suffix(".pdf")
            if not new_pdf.exists():
                old_pdf.rename(new_pdf)
        return new_path

    log.warning("  [slug] target %s already exists, keeping %s", new_path.name, bundle_path.name)
    return bundle_path


def _summarize_blocks(
    blocks: list[dict[str, Any]],
    summarizer: str,
    *,
    title: str = "",
    summary_key: str = "",
) -> list[dict[str, Any]]:
    """Step 6a: Per-block one-line summaries.

    Writes LLM summary into ``block["summaries"][summary_key]``.
    """
    llm = _get_llm(summarizer)
    if llm is None:
        log.warning("  [summarize] no LLM available for %s", summarizer)
        return blocks

    key = summary_key or f"llm:{summarizer}"

    eligible = [
        i
        for i, b in enumerate(blocks)
        if b.get("type") not in _SKIP_EMBED_TYPES
        and len(b.get("text", "").strip()) >= 50
    ]
    log.info("  [summarize] %d/%d blocks eligible", len(eligible), len(blocks))

    done = 0
    errors = 0
    prev_summary = ""
    for idx in eligible:
        block = blocks[idx]
        text = block["text"].strip()

        # Build rolling context: document title + previous summary
        meta_lines = []
        if title:
            meta_lines.append(f"Document: \"{title}\"")
        if prev_summary:
            meta_lines.append(f"Previous paragraph summary: {prev_summary}")
        meta = "\n".join(meta_lines) + "\n" if meta_lines else ""

        try:
            summary = llm(
                f"{meta}"
                f"{_BLOCK_PROMPT_TEMPLATE}\n"
                "Example: 'LOV2 Jα unfolds under blue light; "
                "exposes caging interface for peptide sequences'\n\n"
                f"{text[:2000]}"
            )
            block.setdefault("summaries", {})[key] = summary.strip()
            prev_summary = summary.strip()
            done += 1
            if done % 10 == 0:
                log.info("  [summarize] %d/%d blocks done", done, len(eligible))
        except Exception as exc:
            errors += 1
            log.warning("  [summarize] block %d failed: %s", idx, exc, exc_info=False)

    log.info("  [summarize] blocks done: %d ok, %d errors", done, errors)
    return blocks


def _summarize_paper(
    blocks: list[dict[str, Any]], summarizer: str, *, summary_key: str = ""
) -> str:
    """Step 6b: Paper summary distilled directly from block summaries."""
    llm = _get_llm(summarizer)
    if llm is None:
        return ""

    key = summary_key or f"llm:{summarizer}"
    summaries = [
        b["summaries"][key]
        for b in blocks
        if b.get("summaries", {}).get(key)
    ]
    if not summaries:
        return ""

    combined = "\n".join(f"- {s}" for s in summaries)
    log.info("  [summarize] paper summary from %d block summaries", len(summaries))
    try:
        result = llm(
            f"{_PAPER_PROMPT_TEMPLATE}\n\n{combined[:4000]}"
        ).strip()
        log.info("  [summarize] paper summary done (%d chars)", len(result))
        return result
    except Exception as exc:
        log.warning("  [summarize] paper summary failed: %s", exc)
        return ""


def _embed_blocks(
    blocks: list[dict[str, Any]],
    embedder: Any,
    profile_name: str,
) -> list[dict[str, Any]]:
    """Step 7: Compute embeddings for blocks."""
    texts_to_embed = []
    indices = []

    for i, block in enumerate(blocks):
        if block.get("type") in _SKIP_EMBED_TYPES:
            continue
        text = block.get("text", "").strip()
        if not text:
            continue
        texts_to_embed.append(text)
        indices.append(i)

    if not texts_to_embed:
        return blocks

    try:
        embeddings = embedder(texts_to_embed)
    except Exception:
        return blocks

    for idx, emb in zip(indices, embeddings):
        if "embeddings" not in blocks[idx]:
            blocks[idx]["embeddings"] = {}
        blocks[idx]["embeddings"][profile_name] = emb

    return blocks


def _get_embedder(profile):
    """Get embedding function for a profile config."""
    if profile.provider == "chroma":
        try:
            from chromadb.utils.embedding_functions import (
                DefaultEmbeddingFunction,
            )

            ef = DefaultEmbeddingFunction()

            def _chroma_embed(texts):
                results = ef(texts)
                # Convert numpy arrays to plain lists for JSON serialization
                return [
                    e.tolist() if hasattr(e, "tolist") else list(e) for e in results
                ]

            return _chroma_embed
        except Exception:
            return None

    if profile.provider == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(profile.model)

            def _embed(texts):
                embs = model.encode(texts, normalize_embeddings=True)
                dim = profile.index_dim or profile.dim
                return [e[:dim].tolist() for e in embs]

            return _embed
        except Exception:
            return None

    return None


def _get_llm(summarizer: str):
    """Get a simple LLM callable from a summarizer spec.

    Uses litellm's provider/model format (e.g. "ollama/qwen3.5:9b",
    "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-20250514").

    For ollama models, calls the ollama API directly (litellm drops
    content from thinking models like qwen3.5).

    Returns a function(prompt: str) -> str, or None if unavailable.
    """
    if not summarizer:
        return None

    # Ollama models: call API directly to handle thinking models
    if summarizer.startswith("ollama/"):
        model_name = summarizer[len("ollama/") :]
        return _make_ollama_llm(model_name)

    # All other providers: use litellm
    _ensure_api_keys_in_env()

    try:
        import litellm

        litellm.suppress_debug_info = True

        def _llm_call(prompt: str) -> str:
            resp = litellm.completion(
                model=summarizer,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            return resp.choices[0].message.content or ""

        return _llm_call
    except Exception:
        return None


def _make_ollama_llm(model_name: str):
    """Create a callable that queries ollama directly via HTTP.

    Uses think=false to disable reasoning mode on thinking models
    (massive speedup for simple summarization tasks).
    """
    import httpx

    # Suppress noisy httpx request logging
    logging.getLogger("httpx").setLevel(logging.WARNING)

    def _ollama_call(prompt: str) -> str:
        resp = httpx.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    # Quick connectivity check
    try:
        httpx.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except Exception:
        return None

    return _ollama_call


def _ensure_api_keys_in_env():
    """Push TOML config API keys into standard env vars for litellm.

    If OPENAI_API_KEY or ANTHROPIC_API_KEY is already set in the environment,
    it takes precedence. Only fills in from TOML when the env var is absent.
    """
    import os

    from acatome_meta.config import load_config

    try:
        cfg = load_config(validate=False)
    except Exception:
        return

    _KEY_MAP = {
        "OPENAI_API_KEY": cfg.api.openai_api_key,
        "ANTHROPIC_API_KEY": cfg.api.anthropic_api_key,
    }
    for env_key, value in _KEY_MAP.items():
        if value and not os.environ.get(env_key):
            os.environ[env_key] = value
