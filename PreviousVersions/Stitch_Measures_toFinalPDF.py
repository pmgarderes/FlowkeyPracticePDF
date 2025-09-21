# make_score_pdf_simple.py
# pip install pillow numpy

import os
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageOps

# ---------- helpers ----------
def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))

def page_pixels(page: str = "A4", orientation: str = "portrait", dpi: int = 300) -> Tuple[int, int]:
    sizes_mm = {
        "A4": (210, 297),
        "Letter": (215.9, 279.4),
    }
    w_mm, h_mm = sizes_mm[page]
    if orientation.lower() == "landscape":
        w_mm, h_mm = h_mm, w_mm
    return mm_to_px(w_mm, dpi), mm_to_px(h_mm, dpi)

def otsu_threshold(gray_u8: np.ndarray) -> int:
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = gray_u8.size
    csum = hist.cumsum()
    msum = (hist * np.arange(256)).cumsum()
    denom = csum * (total - csum)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b2 = (msum*total - csum*msum[-1])**2 / np.where(denom == 0, 1, denom)
    return int(np.nanargmax(sigma_b2))

# ---------- main function ----------
def make_pdf_from_measures(
    input_dir: str,
    output_pdf: str,
    measures_per_line: float,       # average target (layout knob)
    zoom: float = 1.0,              # <1 = smaller (zoom out), >1 = larger
    page: str = "A4",
    orientation: str = "portrait",
    dpi: int = 300,
    margin_mm: float = 10.0,
    thr: int | None = None,         # None => Otsu; else fixed 0..255
    min_height_ratio: float = 0.50, # ≥50% black in column
    min_cont_ratio: float = 0.50,   # ≥50% continuous run in that column
):
    """
    1) Load all images (sorted), composite on white, stitch horizontally (same height assumed).
    2) Detect barlines (≥50% black AND ≥50% continuous run), ignoring top/bottom margins.
    3) Wrap only at barlines; scale so ~measures_per_line per row, then apply 'zoom'.
    4) Save paginated PDF.
    """
    # ---- load & stitch on white (no alpha shenanigans) ----
    names = sorted(n for n in os.listdir(input_dir)
                   if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    if not names:
        raise FileNotFoundError(f"No images found in {input_dir}")

    imgs: List[Image.Image] = []
    widths = []
    for n in names:
        im = Image.open(os.path.join(input_dir, n))
        if im.mode in ("RGBA", "LA") or ("A" in im.getbands()):
            bg = Image.new("RGB", im.size, "white")
            alpha = im.split()[-1] if "A" in im.getbands() else None
            bg.paste(im.convert("RGB"), (0, 0), alpha)
            im = bg
        else:
            im = im.convert("RGB")
        imgs.append(im)
        widths.append(im.width)

    H = imgs[0].height  # per your note, heights already match
    stitched_w = int(sum(widths))
    stitched = Image.new("RGB", (stitched_w, H), "white")
    xx = 0
    for im in imgs:
        stitched.paste(im, (xx, 0))
        xx += im.width

    # ---- barline detection (simple + continuous) ----
    g = ImageOps.grayscale(stitched)
    arr = np.asarray(g, dtype=np.uint8)  # (H, W)
    H, W = arr.shape

    if thr is None:
        thr = otsu_threshold(arr)
    bw = arr < thr  # True = black

    # find content band (ignore blank top/bottom margins)
    rows_any = bw.any(axis=1)
    if rows_any.any():
        top = int(np.argmax(rows_any))
        bottom = H - 1 - int(np.argmax(rows_any[::-1]))
    else:
        top, bottom = 0, H - 1
    content_h = max(1, bottom - top + 1)
    band = bw[top:bottom + 1, :]

    # per-column fraction
    col_frac = band.mean(axis=0)

    # per-column max continuous run length (very simple loop; W is fine)
    max_run = np.zeros(W, dtype=int)
    for x in range(W):
        col = band[:, x]
        # compute longest streak of True
        longest = 0
        cur = 0
        for v in col:
            if v:
                cur += 1
                if cur > longest:
                    longest = cur
            else:
                cur = 0
        max_run[x] = longest

    cont_frac = max_run / float(content_h)

    # candidate barline columns
    mask = (col_frac >= min_height_ratio) & (cont_frac >= min_cont_ratio)

    # group contiguous runs => we'll use (start,end) for inclusive barline span
    bar_runs: List[Tuple[int, int]] = []
    in_run = False
    run_start = 0
    for x, m in enumerate(mask):
        if m and not in_run:
            in_run = True
            run_start = x
        elif not m and in_run:
            in_run = False
            bar_runs.append((run_start, x - 1))
    if in_run:
        bar_runs.append((run_start, W - 1))

    # prune very close/very thin runs
    min_bar_width = 1  # keep as 1; we include barline exactly once
    pruned_runs = []
    last_end = -10**9
    for a, b in bar_runs:
        if b - a + 1 >= min_bar_width and (a - last_end) >= 1:
            pruned_runs.append((a, b))
            last_end = b

    # ---- build measures with NO gaps: include barline ONCE with the *right* measure ----
    measures: List[Tuple[int, int]] = []
    prev = 0
    for (a, b) in pruned_runs:
        # left measure ends just BEFORE the barline
        if a > prev:
            measures.append((prev, a - 1))
        # next measure starts AT the barline (includes it)
        prev = a
    # tail
    if prev < W:
        measures.append((prev, W - 1))

    # sanity: drop empty measures
    measures = [(x0, x1) for (x0, x1) in measures if x1 >= x0]

    if not measures:
        # fallback: one big measure
        measures = [(0, W - 1)]

    # ---- layout: wrap only at measures, with zoom control ----
    page_w, page_h = page_pixels(page, orientation, dpi)
    margin_px = mm_to_px(margin_mm, dpi)
    content_w = page_w - 2 * margin_px
    content_h = page_h - 2 * margin_px

    # no horizontal gaps at all
    col_spacing = 0
    row_spacing = mm_to_px(3.0, dpi)  # small vertical gap between lines

    # target average scale from measures_per_line, then apply zoom
    avg_w = float(np.mean([(x1 - x0 + 1) for (x0, x1) in measures]))
    base_scale = (content_w - col_spacing * (measures_per_line - 1)) / (avg_w * measures_per_line)
    scale = max(0.05, min(5.0, base_scale * zoom))
    target_h = int(round(H * scale))

    # paginate
    pages: List[Image.Image] = []
    page_img = Image.new("RGB", (page_w, page_h), "white")
    x = margin_px
    y = margin_px
    row_h = 0

    def new_page():
        nonlocal page_img, x, y, row_h
        pages.append(page_img)
        page_img = Image.new("RGB", (page_w, page_h), "white")
        x = margin_px
        y = margin_px
        row_h = 0

    for (x0, x1) in measures:
        crop = stitched.crop((x0, 0, x1 + 1, H))  # +1 because right is exclusive in PIL
        w_scaled = int(round(crop.width * scale))
        if w_scaled <= 0 or target_h <= 0:
            continue
        blk = crop.resize((w_scaled, target_h), Image.BICUBIC)

        # wrap at page edge (no extra spacing)
        if x > margin_px and x + blk.width > margin_px + content_w:
            y += row_h + row_spacing
            x = margin_px
            row_h = 0

        # new page if needed
        if y + blk.height > margin_px + content_h:
            new_page()

        page_img.paste(blk, (x, y))
        x += blk.width + col_spacing
        row_h = max(row_h, blk.height)

    if x != margin_px or y != margin_px:
        pages.append(page_img)
    if not pages:
        pages = [Image.new("RGB", (page_w, page_h), "white")]

    first, rest = pages[0], pages[1:]
    first.save(output_pdf, save_all=True, append_images=rest)
    print(f"[done] measures={len(measures)} pages={len(pages)} scale={scale:.3f} pdf={output_pdf}")


# ---------- example ----------
if __name__ == "__main__":
    make_pdf_from_measures(
        input_dir=r"C:/Users/pmgar/OneDrive/Documents/FlowKey/LettreAElise/Images",
        output_pdf = r"C:/Users/pmgar/OneDrive/Documents/FlowKey/LettreAElise/Elvis_stitched_A4.pdf",
        measures_per_line=4.0,  # average measures  per line
        zoom=1.3,              # try 0.7 (smaller) .. 1.0 .. 1.3 (larger)
        page="A4",
        orientation="portrait",
        dpi=300,
    )

