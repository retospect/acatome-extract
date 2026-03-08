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
from pathlib import Path
from typing import Any

log = logging.getLogger("acatome_extract.enrich")

from acatome_extract.bundle import read_bundle, write_bundle

# Block types that skip embeddings
_SKIP_EMBED_TYPES = {"section_header", "title", "author", "equation"}


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

    # Step 6: Summarize
    if summarize:
        title = data.get("header", {}).get("title", "")
        blocks = _summarize_blocks(blocks, summarizer, title=title)
        paper_summary = _summarize_paper(blocks, summarizer)
        data["enrichment_meta"] = data.get("enrichment_meta") or {}
        data["enrichment_meta"]["summarizer"] = summarizer
        data["enrichment_meta"]["paper_summary"] = paper_summary

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


def _summarize_blocks(
    blocks: list[dict[str, Any]], summarizer: str, *, title: str = ""
) -> list[dict[str, Any]]:
    """Step 6a: Per-block one-line summaries."""
    llm = _get_llm(summarizer)
    if llm is None:
        log.warning("  [summarize] no LLM available for %s", summarizer)
        return blocks

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
                "Terse telegram-style summary, one line. "
                "No articles, no filler ('This passage', 'The authors'). "
                "Core claim first; detail after semicolon. "
                "Must make sense if truncated at semicolon.\n"
                "Example: 'LOV2 Jα unfolds under blue light; "
                "exposes caging interface for peptide sequences'\n\n"
                f"{text[:2000]}"
            )
            block["summary"] = summary.strip()
            prev_summary = block["summary"]
            done += 1
            if done % 10 == 0:
                log.info("  [summarize] %d/%d blocks done", done, len(eligible))
        except Exception as exc:
            errors += 1
            log.warning("  [summarize] block %d failed: %s", idx, exc, exc_info=False)

    log.info("  [summarize] blocks done: %d ok, %d errors", done, errors)
    return blocks


def _summarize_paper(blocks: list[dict[str, Any]], summarizer: str) -> str:
    """Step 6b: Paper summary distilled directly from block summaries."""
    llm = _get_llm(summarizer)
    if llm is None:
        return ""

    summaries = [b["summary"] for b in blocks if b.get("summary")]
    if not summaries:
        return ""

    combined = "\n".join(f"- {s}" for s in summaries)
    log.info("  [summarize] paper summary from %d block summaries", len(summaries))
    try:
        result = llm(
            f"3-5 telegram-style lines, broadest claim to finest detail. "
            f"No articles, no filler. Line 1 stands alone as summary. "
            f"Each subsequent line narrows scope.\n\n{combined[:4000]}"
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
