# extract_flowkey_local_windows_safe.py
# Parse a locally saved HTML page and export sheet images.
# Works on Windows paths; uses pathname2url/url2pathname for robust file:// handling.

import os
import re
import base64
import shutil
from html import unescape
from urllib.parse import urljoin, urlparse
from urllib.request import pathname2url, url2pathname

import requests
from bs4 import BeautifulSoup


def extract_sheet_images_from_html(
    html_file: str,
    out_dir: str,
    image_selector: str = ".sheet-container .scrollable-sheet .sheet-image",
    sort_by_top: bool = True,       # set False to keep document order as-is
):
    r"""
            Args:
            html_file: Path to your saved .html (e.g., "C:\Users\you\Downloads\player.html")
            out_dir: Output directory to write images
            image_selector: CSS selector for the image tiles (".sheet-image" usually works)
            sort_by_top: If True, order files by the CSS 'top: Npx' value (Flowkey pages)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Base URL for resolving relative paths (Windows-safe)
    html_dir = os.path.dirname(os.path.abspath(html_file))
    file_base = "file:///" + pathname2url(html_dir) + "/"

    # Load HTML
    with open(html_file, "rb") as f:
        soup = BeautifulSoup(f, "html.parser")

    elements = soup.select(image_selector)
    print(f"[info] Found {len(elements)} elements with selector '{image_selector}'")

    entries = []  # (index, top_val, url_raw)
    for idx, el in enumerate(elements, 1):
        style = unescape(el.get("style") or "")
        # Try CSS background-image
        m_bg = re.search(r'background-image\s*:\s*url\((["\']?)(.*?)\1\)', style, flags=re.I)
        url_raw = None
        if m_bg and m_bg.group(2):
            url_raw = m_bg.group(2)
        else:
            # Fallback <img src=...>
            img = el.find("img")
            if img and img.get("src"):
                url_raw = unescape(img.get("src"))

        if not url_raw:
            continue

        # Extract numeric 'top: Npx' if present
        m_top = re.search(r'\btop\s*:\s*([-+]?\d+(\.\d+)?)px', style, flags=re.I)
        top_val = float(m_top.group(1)) if m_top else float(idx)  # fallback to DOM order
        entries.append((idx, top_val, url_raw))

    if not entries:
        print("[warn] No URLs found. Make sure you saved after scrolling through all pages, and selector matches the tiles.")
        return

    # Sort if requested (by top, then by index to stabilize ties)
    if sort_by_top:
        entries.sort(key=lambda t: (t[1], t[0]))

    # Deduplicate by final resolved URL while preserving the chosen order
    resolved_entries = []
    seen = set()

    for idx, top_val, raw in entries:
        u = (raw or "").strip().strip('"').strip("'")
        u = unescape(u)

        if u.startswith("data:"):
            resolved = u  # keep as-is
        else:
            parsed = urlparse(u)
            if not parsed.scheme:
                # Resolve relative/local paths against the HTML directory
                resolved = urljoin(file_base, u)
            else:
                resolved = u

        if resolved not in seen:
            seen.add(resolved)
            resolved_entries.append((idx, top_val, raw, resolved))

    print(f"[info] Extracted {len(resolved_entries)} unique image references")

    # Helpers
    def guess_ext(u: str) -> str:
        if u.startswith("data:"):
            m = re.search(r"data:image/(png|jpeg|jpg|webp|gif|svg)\b", u, re.I)
            if m:
                ext = m.group(1).lower().replace("jpeg", "jpg")
                return "." + ext
            return ".png"
        name = os.path.basename(urlparse(u).path).lower()
        m = re.search(r"\.(png|jpg|jpeg|webp|gif|svg)$", name)
        return "." + (m.group(1) if m else "jpg")

    def save_http(u: str, path: str):
        r = requests.get(u, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

    def save_file(u: str, path: str):
        # u is expected to be an absolute file URL like file:///C:/...
        parsed = urlparse(u)
        src_path = url2pathname(parsed.path)  # -> Windows path C:\...
        if not os.path.isabs(src_path):
            # Just in case the saved HTML used relative path without scheme
            src_path = os.path.join(html_dir, src_path)
        shutil.copyfile(src_path, path)

    def save_data_url(u: str, path: str):
        head, b64 = u.split(",", 1)
        data = base64.b64decode(b64)
        with open(path, "wb") as f:
            f.write(data)

    # Save images
    for i, (_, top_val, raw_u, resolved_u) in enumerate(resolved_entries, 1):
        ext = guess_ext(resolved_u)
        # Include the 'top' position in filename if you like (helps debug order)
        out_name = f"sheet_{i:03d}{ext}"  # or f"sheet_{i:03d}_top{int(round(top_val)):05d}{ext}"
        out_path = os.path.join(out_dir, out_name)
        print(f"[save] {i}/{len(resolved_entries)} -> {out_name}")

        try:
            if resolved_u.startswith("data:"):
                save_data_url(resolved_u, out_path)
            else:
                scheme = urlparse(resolved_u).scheme.lower()
                if scheme in ("http", "https"):
                    save_http(resolved_u, out_path)
                elif scheme == "file":
                    save_file(resolved_u, out_path)
                else:
                    # Last resort: treat as local path relative to HTML dir
                    local_guess = os.path.join(html_dir, raw_u)
                    shutil.copyfile(local_guess, out_path)
        except Exception as e:
            print(f"[error] {resolved_u} -> {e}")

    print("[done] Saved", len(resolved_entries), "images to", out_dir)




extract_sheet_images_from_html(
    html_file=r"C:/Users/pmgar/OneDrive/Documents/FlowKey/LettreAElise/La Lettre à Élise _ flowkey.html",
    out_dir=r"C:/Users/pmgar/OneDrive/Documents/FlowKey/LettreAElise",
    image_selector=".sheet-container .scrollable-sheet .sheet-image",
    sort_by_top = True
)

