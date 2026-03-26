"""Open paper PDFs in the system viewer.

Resolves a paper slug to its PDF path via the store's bundle_path,
then opens it in Preview (macOS) with optional page navigation.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def resolve_pdf(slug: str) -> Path:
    """Resolve a paper slug to its PDF path.

    The PDF sits next to the .acatome bundle with a .pdf extension.

    Raises:
        FileNotFoundError: If paper not in store or PDF file missing.
    """
    from acatome_store.store import Store

    store = Store()
    paper = store.get(slug)
    if paper is None:
        raise FileNotFoundError(f"Paper not found: {slug}")

    bundle_path = paper.get("bundle_path")
    if not bundle_path:
        raise FileNotFoundError(f"No bundle_path for {slug}")

    pdf_path = Path(bundle_path).with_suffix(".pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    return pdf_path


def open_pdf(pdf_path: Path, page: int | None = None) -> str:
    """Open a PDF in the system viewer, optionally at a specific page.

    On macOS, uses Preview with AppleScript for page navigation.

    Returns:
        Status message string.
    """
    if sys.platform != "darwin":
        subprocess.Popen(["xdg-open", str(pdf_path)])
        return f"Opened {pdf_path.name}" + (f" (page {page} — manual navigation needed)" if page else "")

    # macOS: open in Preview
    subprocess.Popen(["open", "-a", "Preview", str(pdf_path)])

    if page and page > 1:
        # Give Preview a moment to open/focus the document
        time.sleep(0.8)
        # Use AppleScript to trigger Go To Page (Cmd+Option+G) and type the page number
        script = f'''
        tell application "Preview" to activate
        delay 0.3
        tell application "System Events"
            keystroke "g" using {{command down, option down}}
            delay 0.3
            keystroke "{page}"
            delay 0.1
            keystroke return
        end tell
        '''
        subprocess.Popen(["osascript", "-e", script])

    msg = f"Opened {pdf_path.name}"
    if page:
        msg += f" at page {page}"
    return msg


def open_paper(slug: str, page: int | None = None) -> str:
    """Resolve slug and open the PDF. Returns status message."""
    pdf_path = resolve_pdf(slug)
    return open_pdf(pdf_path, page)
