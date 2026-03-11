"""Directory watcher: PDF appears → extract → enrich → ingest.

Monitors a directory for new PDF files and runs the full pipeline
automatically. Processed PDFs move to ``completed/``, failures to
``errors/`` (both inside the watch directory).
"""

from __future__ import annotations

import getpass
import hashlib
import logging
import shutil
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock
from typing import Any

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

log = logging.getLogger("acatome.watch")

# Minimum time (seconds) a file must be stable (no size change) before processing
DEFAULT_DEBOUNCE = 3.0
DEFAULT_POLL_INTERVAL = 1.0


def _pdf_hash(path: Path) -> str:
    """SHA-256 hash of a PDF file (matches store dedup hash)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class UnverifiedPaperError(Exception):
    """Raised when a paper fails metadata verification."""

    def __init__(self, message: str, bundle_path: Path | None = None):
        super().__init__(message)
        self.bundle_path = bundle_path


class _PdfHandler(FileSystemEventHandler):
    """Watchdog handler that queues new PDFs for processing."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            self._callback(Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        dest = getattr(event, "dest_path", None)
        if dest and not event.is_directory and dest.lower().endswith(".pdf"):
            self._callback(Path(dest))


def _wait_stable(path: Path, debounce: float) -> bool:
    """Wait until file size stops changing. Returns False if file disappears."""
    prev_size = -1
    while True:
        if not path.exists():
            return False
        size = path.stat().st_size
        if size == prev_size and size > 0:
            return True
        prev_size = size
        time.sleep(debounce)


def _ts() -> str:
    """Compact timestamp for log output."""
    return datetime.now().strftime("%H:%M:%S")


def _validate_shared_bundle(pdf_path: Path, bundle_path: Path) -> dict[str, Any] | None:
    """Validate a pre-existing .acatome bundle against its PDF.

    Returns bundle header dict if valid, None if invalid.
    Checks: readable, pdf_hash matches, title exists.
    """
    from acatome_extract.bundle import read_bundle

    try:
        data = read_bundle(bundle_path)
    except Exception as exc:
        log.warning("  [shared] cannot read %s: %s", bundle_path.name, exc)
        return None

    header = data.get("header", {})

    # pdf_hash must match the actual PDF
    actual_hash = _pdf_hash(pdf_path)
    bundle_hash = header.get("pdf_hash", "")
    if not bundle_hash or actual_hash != bundle_hash:
        log.warning(
            "  [shared] hash mismatch: bundle=%s pdf=%s",
            bundle_hash[:12],
            actual_hash[:12],
        )
        return None

    # Must have a title
    if not header.get("title"):
        log.warning("  [shared] bundle has no title — rejecting")
        return None

    log.info("  [shared] valid bundle: %s", bundle_path.name)
    return header


def run_pipeline(
    pdf_path: Path,
    *,
    output_dir: Path | None = None,
    enrich: bool = True,
    summarize: bool = True,
    summarizer: str = "",
    ingest: bool = True,
) -> dict[str, Any]:
    """Run full pipeline on a single PDF. Returns result dict.

    If a .acatome bundle exists alongside the PDF (same stem),
    validates it and skips extraction/enrichment (shared bundle fast path).
    """
    from acatome_extract.bundle import read_bundle
    from acatome_extract.pipeline import extract

    result: dict[str, Any] = {"pdf": str(pdf_path)}

    # --- shared bundle fast path ---
    companion = pdf_path.with_suffix(".acatome")
    shared_header = None
    if companion.is_file():
        shared_header = _validate_shared_bundle(pdf_path, companion)

    if shared_header is not None:
        # Copy bundle to output_dir (if different from source)
        from acatome_meta.config import ACATOME_HOME

        dest_dir = output_dir or Path(ACATOME_HOME) / "papers"
        slug = shared_header.get("slug", pdf_path.stem)
        first_letter = slug[0].lower() if slug else "_"
        final_dir = dest_dir / first_letter
        final_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = final_dir / f"{slug}.acatome"
        if companion.resolve() != bundle_path.resolve():
            shutil.copy2(companion, bundle_path)
        result["bundle"] = str(bundle_path)
        result["slug"] = slug
        result["doi"] = shared_header.get("doi") or ""
        result["title"] = shared_header.get("title") or ""
        result["verified"] = shared_header.get("verified", False)
        result["shared"] = True
        log.info("  [shared] skipping extract+enrich → %s", bundle_path.name)
        header = shared_header
    else:
        # --- extract ---
        log.info("  [extract] parsing PDF...")
        bundle_path = extract(pdf_path, output_dir=output_dir)
        result["bundle"] = str(bundle_path)
        result["slug"] = bundle_path.stem
        log.info("  [extract] → %s", bundle_path.name)

        header = {}
        try:
            header = read_bundle(bundle_path).get("header", {})
            result["doi"] = header.get("doi") or ""
            result["title"] = header.get("title") or ""
            result["verified"] = header.get("verified", False)
        except Exception:
            result["doi"] = ""
            result["title"] = ""
            result["verified"] = False

    # Reject unverified papers — require manual review
    reject_reason = _check_rejection(result, header)
    if reject_reason:
        raise UnverifiedPaperError(
            f"{reject_reason} "
            f"Slug: {result.get('slug', '?')}, "
            f"Title: {result.get('title', '?')!r}. "
            f"Move to inbox manually after review, or ingest with: "
            f"acatome-store ingest {bundle_path}",
            bundle_path=bundle_path,
        )

    log.info(
        "  [verified] %s — %s",
        result.get("doi") or "no DOI",
        (result.get("title") or "untitled")[:60],
    )

    # --- enrich (skip for shared bundles — already enriched) ---
    if enrich and not result.get("shared"):
        from acatome_extract.enrich import enrich as do_enrich
        from acatome_meta.config import load_config

        log.info("  [enrich] embeddings + summaries...")
        cfg = load_config()
        sm = summarizer or cfg.extract.enrich.summarizer
        do_enrich(bundle_path, profiles=["default"], summarize=summarize, summarizer=sm)
        result["enriched"] = True
        log.info("  [enrich] done")

    # --- ingest ---
    if ingest:
        try:
            from acatome_store.store import Store
        except ImportError:
            log.warning("acatome-store not installed — skipping ingest")
            result["ingested"] = False
            return result

        log.info("  [ingest] storing...")
        store = Store()
        ref_id = store.ingest(bundle_path)
        store.close()
        result["ref_id"] = ref_id
        result["ingested"] = True
        log.info("  [ingest] ref_id=%d", ref_id)

    return result


def watch(
    watch_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    recursive: bool = True,
    backfill: bool = True,
    enrich: bool = True,
    summarize: bool = False,
    summarizer: str = "",
    ingest: bool = True,
    keep: bool = False,
    user: str = "",
    debounce: float = DEFAULT_DEBOUNCE,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    use_polling: bool = False,
) -> None:
    """Watch a directory and auto-process PDFs.

    Args:
        watch_dir: Directory to monitor.
        output_dir: Bundle output directory (default: ~/.acatome/papers/).
        recursive: Watch subdirectories (default True).
        backfill: Process existing PDFs on startup.
        enrich: Run enrichment (embeddings). LLM summaries off by default.
        summarize: Run LLM summaries during enrichment (default False).
        summarizer: litellm model spec for enrichment.
        ingest: Ingest into store after extraction.
        keep: Don't move PDFs after processing (leave in place).
        debounce: Seconds to wait for file stability.
        poll_interval: Polling interval for fallback observer.
        use_polling: Force polling observer (for network mounts).
        user: User name for ingest log attribution (default: OS user).
    """
    user = user or getpass.getuser()
    watch_dir = Path(watch_dir).resolve()
    if not watch_dir.is_dir():
        raise FileNotFoundError(f"Watch directory not found: {watch_dir}")

    completed_dir = watch_dir / "completed"
    errors_dir = watch_dir / "errors"
    duplicates_dir = errors_dir / "duplicates"
    completed_dir.mkdir(exist_ok=True)
    errors_dir.mkdir(exist_ok=True)
    duplicates_dir.mkdir(exist_ok=True)

    stop_event = Event()
    processing_lock = Lock()

    # Graceful shutdown
    def _signal_handler(signum, frame):
        log.info("shutting down (finishing current)")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Track processed files and hashes to avoid double-processing
    seen: set[Path] = set()
    seen_hashes: set[str] = set()

    def _should_skip(pdf: Path) -> bool:
        """Skip files in completed/, errors/, or duplicates/ subdirs, or already seen."""
        resolved = pdf.resolve()
        for skip_dir in (completed_dir, errors_dir):
            try:
                resolved.relative_to(skip_dir.resolve())
                return True
            except ValueError:
                pass
        return resolved in seen

    def _process(pdf: Path) -> None:
        if _should_skip(pdf):
            return

        with processing_lock:
            # Re-check after acquiring lock
            if _should_skip(pdf) or stop_event.is_set():
                return
            seen.add(pdf.resolve())

        log.info(f"new: {pdf.name}")

        # Wait for file to finish writing
        if not _wait_stable(pdf, debounce):
            log.warning(f"  file disappeared: {pdf.name}")
            return

        # Duplicate check by SHA-256
        filehash = _pdf_hash(pdf)
        if filehash in seen_hashes:
            log.info(f"  duplicate (hash): {pdf.name} → errors/duplicates/")
            if not keep:
                _move_to(pdf, duplicates_dir)
            return
        seen_hashes.add(filehash)

        t0 = time.monotonic()
        try:
            log.info("  extracting...")
            result = run_pipeline(
                pdf,
                output_dir=output_dir,
                enrich=enrich,
                summarize=summarize,
                summarizer=summarizer,
                ingest=ingest,
            )

            slug = result.get("slug", pdf.stem)
            elapsed = time.monotonic() - t0

            parts = []
            if result.get("shared"):
                parts.append(f"shared → {slug}.acatome")
            else:
                parts.append(f"extract → {slug}.acatome")
            if result.get("enriched"):
                parts.append("enriched")
            if result.get("ingested"):
                parts.append(f"ref_id={result.get('ref_id')}")
            log.info(f"  ✓ {', '.join(parts)} ({elapsed:.0f}s)")

            # Log to completed/ingest.log
            _log_completed(completed_dir, user, result, pdf)

            # Move to completed
            if not keep:
                _move_to(pdf, completed_dir)
                # Move companion .acatome from inbox (shared bundles)
                companion = pdf.with_suffix(".acatome")
                if companion.is_file():
                    _move_to(companion, completed_dir)

        except Exception as e:
            elapsed = time.monotonic() - t0
            log.error(f"  ✗ {pdf.name}: {e} ({elapsed:.0f}s)")
            _write_error(errors_dir, pdf, e)
            if not keep:
                _move_to(pdf, errors_dir)
            # Move orphan bundle out of papers/ on verification failure
            bp = getattr(e, "bundle_path", None)
            if bp and isinstance(bp, Path) and bp.exists():
                _move_to(bp, errors_dir)
                log.info(f"  moved orphan bundle to errors/: {bp.name}")

    # --- Backfill (newest first) ---
    if backfill:
        if recursive:
            existing = list(watch_dir.rglob("*.pdf"))
        else:
            existing = list(watch_dir.glob("*.pdf"))
        # Sort by modification time, newest first
        existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if existing:
            log.info(f"backfill: {len(existing)} existing PDF(s) (newest first)")
            for i, pdf in enumerate(existing, 1):
                if stop_event.is_set():
                    break
                log.info(f"[{i}/{len(existing)}] {pdf.name}")
                _process(pdf)

    # --- Start watcher ---
    handler = _PdfHandler(_process)
    ObserverClass = PollingObserver if use_polling else Observer
    observer = ObserverClass()
    if use_polling:
        observer = PollingObserver(timeout=poll_interval)
    observer.schedule(handler, str(watch_dir), recursive=recursive)
    observer.start()
    log.info(f"watching {watch_dir} {'(recursive)' if recursive else ''} ...")

    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    finally:
        observer.stop()
        observer.join()
        log.info("stopped")


def _check_rejection(result: dict[str, Any], header: dict[str, Any]) -> str:
    """Return a rejection reason string, or empty string if paper is OK.

    Rejects papers that:
    1. Failed metadata verification (verified=False)
    2. Have garbage metadata: no DOI/arxiv AND (no title or anon author slug)
    """
    # Gate 1: explicit verification failure
    if not result.get("verified"):
        warnings = header.get("verify_warnings", [])
        warn_str = "; ".join(warnings) if warnings else "no DOI/title match"
        return f"Paper failed verification ({warn_str})."

    # Gate 2: garbage metadata — no identifier + no real title
    has_doi = bool(result.get("doi"))
    has_arxiv = bool(header.get("arxiv_id"))
    has_s2 = bool(header.get("s2_id"))
    has_identifier = has_doi or has_arxiv or has_s2

    title = (result.get("title") or "").strip().lower()
    slug = (result.get("slug") or "").lower()

    title_is_garbage = (
        not title
        or title == "untitled"
        or title.startswith("pii:")
        or title.startswith("pii ")
        or len(title) < 5
    )
    slug_is_anon = slug.startswith("anon")

    if not has_identifier and (title_is_garbage or slug_is_anon):
        return (
            f"Paper has no DOI/arxiv/s2 identifier and looks like garbage metadata "
            f"(title={title!r}, slug={slug!r})."
        )

    return ""


def _move_to(src: Path, dest_dir: Path) -> Path:
    """Move file to dest_dir, handling name conflicts."""
    dest = dest_dir / src.name
    if dest.exists():
        stem = src.stem
        suffix = src.suffix
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        dest = dest_dir / f"{stem}_{ts}{suffix}"
    shutil.move(str(src), str(dest))
    return dest


def _log_completed(
    completed_dir: Path, user: str, result: dict[str, Any], pdf: Path
) -> None:
    """Append a greppable TSV line to completed/ingest.log.

    Format: timestamp  user  slug  doi  filename  ref_id  title
    Grep example: grep 'reto' completed/ingest.log
    """
    log_file = completed_dir / "ingest.log"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    slug = result.get("slug", "")
    doi = result.get("doi", "")
    ref_id = result.get("ref_id", "")
    title = result.get("title", "")
    line = f"{ts}\t{user}\t{slug}\t{doi}\t{pdf.name}\t{ref_id}\t{title}\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)


def _write_error(errors_dir: Path, pdf: Path, error: Exception) -> None:
    """Write error details next to the failed PDF."""
    error_file = errors_dir / f"{pdf.stem}.error.txt"
    error_file.write_text(
        f"PDF: {pdf.name}\n"
        f"Time: {datetime.now(timezone.utc).isoformat()}\n"
        f"Error: {error}\n\n"
        f"Traceback:\n{traceback.format_exc()}"
    )
