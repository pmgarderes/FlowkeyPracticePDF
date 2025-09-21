# flowkey2pdf_gui.py
# pip install pillow numpy beautifulsoup4 requests

import os, re, base64, shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from html import unescape
from urllib.parse import urljoin, urlparse
from urllib.request import pathname2url, url2pathname

import requests
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image, ImageOps

from io import BytesIO
from PIL import Image

# -------------------- HTML -> images --------------------

def extract_sheet_images_from_html(
    html_file: str,
    out_dir: str,
    image_selector: str = ".sheet-container .scrollable-sheet .sheet-image",
    sort_by_top: bool = True
) -> int:
    """Parse a saved Flowkey-like HTML and export sheet images to out_dir. Returns count saved."""
    os.makedirs(out_dir, exist_ok=True)
    html_dir = os.path.dirname(os.path.abspath(html_file))
    file_base = "file:///" + pathname2url(html_dir) + "/"

    with open(html_file, "rb") as f:
        soup = BeautifulSoup(f, "html.parser")

    elements = soup.select(image_selector)
    entries = []  # (idx, top_val, url_raw)
    for idx, el in enumerate(elements, 1):
        style = unescape(el.get("style") or "")
        m_bg = re.search(r'background-image\s*:\s*url\((["\']?)(.*?)\1\)', style, flags=re.I)
        url_raw = None
        if m_bg and m_bg.group(2):
            url_raw = m_bg.group(2)
        else:
            img = el.find("img")
            if img and img.get("src"):
                url_raw = unescape(img.get("src"))
        if not url_raw:
            continue
        m_top = re.search(r'\btop\s*:\s*([-+]?\d+(?:\.\d+)?)px', style, flags=re.I)
        top_val = float(m_top.group(1)) if m_top else float(idx)
        entries.append((idx, top_val, url_raw))

    if not entries:
        return 0

    if sort_by_top:
        entries.sort(key=lambda t: (t[1], t[0]))

    seen, resolved = set(), []
    for idx, top_val, raw in entries:
        u = (raw or "").strip().strip('"').strip("'")
        u = unescape(u)
        if u.startswith("data:"):
            final = u
        else:
            parsed = urlparse(u)
            final = u if parsed.scheme else urljoin(file_base, u)
        if final not in seen:
            seen.add(final)
            resolved.append((idx, top_val, raw, final))

    def guess_ext(u: str) -> str:
        if u.startswith("data:"):
            m = re.search(r"data:image/(png|jpeg|jpg|webp|gif|svg)\b", u, re.I)
            ext = (m.group(1).lower().replace("jpeg", "jpg") if m else "png")
            return "." + ext
        name = os.path.basename(urlparse(u).path).lower()
        m = re.search(r"\.(png|jpg|jpeg|webp|gif|svg)$", name)
        return "." + (m.group(1) if m else "jpg")

    def save_http(u: str, path: str):
        r = requests.get(u, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

    def save_file(u: str, path: str):
        parsed = urlparse(u)
        src_path = url2pathname(parsed.path)
        if not os.path.isabs(src_path):
            src_path = os.path.join(html_dir, src_path)
        shutil.copyfile(src_path, path)

    def save_data(u: str, path: str):
        head, b64 = u.split(",", 1)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))

    count = 0
    for i, (_, top_val, raw_u, final) in enumerate(resolved, 1):
        ext = guess_ext(final)
        out_name = f"sheet_{i:03d}{ext}"
        out_path = os.path.join(out_dir, out_name)
        try:
            if final.startswith("data:"):
                save_data(final, out_path)
            else:
                scheme = urlparse(final).scheme.lower()
                if scheme in ("http", "https"):
                    save_http(final, out_path)
                elif scheme == "file":
                    save_file(final, out_path)
                else:
                    shutil.copyfile(os.path.join(html_dir, raw_u), out_path)
            count += 1
        except Exception as e:
            print("[error]", final, "->", e)
    return count

# -------------------- images -> PDF --------------------

def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))

def page_pixels(page: str = "A4", orientation: str = "portrait", dpi: int = 300):
    sizes_mm = {"A4": (210, 297), "Letter": (215.9, 279.4)}
    w_mm, h_mm = sizes_mm[page]
    if orientation.lower() == "landscape":
        w_mm, h_mm = h_mm, w_mm
    return mm_to_px(w_mm, dpi), mm_to_px(h_mm, dpi)

def make_pdf_from_measures(
    input_dir: str,
    output_pdf: str,
    measures_per_line: float,
    zoom: float = 1.0,
    page: str = "A4",
    orientation: str = "portrait",
    dpi: int = 300,
    margin_mm: float = 10.0,
    thr: int = 10,              # FIXED default threshold (matches your observation)
    min_height_ratio: float = 0.50,
    min_cont_ratio: float = 0.50,
):
    # Load & stitch horizontally (composite on white)
    names = sorted(n for n in os.listdir(input_dir)
                   if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    if not names:
        raise FileNotFoundError(f"No images found in {input_dir}")

    imgs, widths = [], []
    for n in names:
        im = Image.open(os.path.join(input_dir, n))
        if im.mode in ("RGBA", "LA") or ("A" in im.getbands()):
            bg = Image.new("RGB", im.size, "white")
            alpha = im.split()[-1] if "A" in im.getbands() else None
            bg.paste(im.convert("RGB"), (0, 0), alpha)
            im = bg
        else:
            im = im.convert("RGB")
        imgs.append(im); widths.append(im.width)

    H = imgs[0].height
    W = int(sum(widths))
    stitched = Image.new("RGB", (W, H), "white")
    xx = 0
    for im in imgs:
        stitched.paste(im, (xx, 0)); xx += im.width

    # ---- Detect barlines on stitched image ----
    g = ImageOps.grayscale(stitched)
    arr = np.asarray(g, dtype=np.uint8)           # (H, W)

    # fixed threshold (works best on these assets)
    bw = (arr < thr)

    # ignore blank top/bottom margins
    rows_any = bw.any(axis=1)
    if rows_any.any():
        top = int(np.argmax(rows_any))
        bottom = H - 1 - int(np.argmax(rows_any[::-1]))
    else:
        top, bottom = 0, H - 1
    content_h = max(1, bottom - top + 1)
    band = bw[top:bottom+1, :]

    # column black fraction
    col_frac = band.mean(axis=0)

    # longest continuous black run per column (strict continuity)
    max_run = np.zeros(W, dtype=int)
    for xi in range(W):
        col = band[:, xi]
        longest = cur = 0
        for v in col:
            if v: cur += 1; longest = max(longest, cur)
            else: cur = 0
        max_run[xi] = longest
    cont_frac = max_run / float(content_h)

    # qualify as barline
    mask = (col_frac >= min_height_ratio) & (cont_frac >= min_cont_ratio)

    # contiguous True runs -> barline spans
    runs = []
    in_run = False; start = 0
    for xi, m in enumerate(mask):
        if m and not in_run:
            in_run = True; start = xi
        elif not m and in_run:
            in_run = False; runs.append((start, xi-1))
    if in_run: runs.append((start, W-1))

    # measures with NO gaps: barline belongs to the RIGHT measure
    measures = []
    prev = 0
    for a, b in runs:
        if a > prev:
            measures.append((prev, a-1))
        prev = a
    if prev < W: measures.append((prev, W-1))
    measures = [(x0, x1) for (x0, x1) in measures if x1 >= x0]
    if not measures: measures = [(0, W-1)]

    # ---- Layout: scale whole stitched image ONCE → crop from scaled image (no seams) ----
    page_w, page_h = page_pixels(page, orientation, dpi)
    margin_px = mm_to_px(margin_mm, dpi)
    content_w = page_w - 2*margin_px
    content_h_px = page_h - 2*margin_px

    # no horizontal gaps
    col_spacing = 0
    row_spacing = mm_to_px(3.0, dpi)

    # compute scale from target measures_per_line, then apply zoom
    avg_w = float(np.mean([(x1-x0+1) for (x0,x1) in measures]))
    base_scale = (content_w - col_spacing*(measures_per_line-1)) / (avg_w*measures_per_line)
    scale = max(0.05, min(5.0, base_scale * zoom))

    # scale entire stitched image once
    W_scaled = int(round(W * scale))
    H_scaled = int(round(H * scale))
    # scaled = stitched.resize((W_scaled, H_scaled), Image.BICUBIC)
    scaled = stitched.resize((W_scaled, H_scaled), Image.BICUBIC).convert("L").convert("RGB")

    # compute scaled measure widths (sum to W_scaled), then crop from 'scaled'
    scaled_measures = []
    cursor = 0
    for (x0, x1) in measures:
        w = (x1 - x0 + 1)
        w_s = max(1, int(round(w * scale)))
        scaled_measures.append((cursor, cursor + w_s))
        cursor += w_s
    # last one correction to hit exact end (avoid any 1px leftover)
    if scaled_measures:
        s0, s1 = scaled_measures[-1]
        scaled_measures[-1] = (s0, W_scaled)

    # paginate
    pages = []
    page_img = Image.new("RGB", (page_w, page_h), "white")
    x = margin_px; y = margin_px; row_h = 0

    def new_page():
        nonlocal page_img, x, y, row_h
        pages.append(page_img)
        page_img = Image.new("RGB", (page_w, page_h), "white")
        x = margin_px; y = margin_px; row_h = 0

    for (sx0, sx1) in scaled_measures:
        crop = scaled.crop((sx0, 0, sx1, H_scaled))
        # wrap at edge
        if x > margin_px and x + crop.width > margin_px + content_w:
            y += row_h + row_spacing
            x = margin_px; row_h = 0
        if y + crop.height > margin_px + content_h_px:
            new_page()

        page_img.paste(crop, (x, y))
        x += crop.width + col_spacing
        row_h = max(row_h, crop.height)

    if x != margin_px or y != margin_px: pages.append(page_img)
    if not pages: pages = [Image.new("RGB", (page_w, page_h), "white")]

    first, rest = pages[0], pages[1:]
    first.save(output_pdf, save_all=True, append_images=rest)
    return len(measures), len(pages), scale

# -------------------- Tiny GUI --------------------

class App:
    def __init__(self, root):
        self.root = root
        root.title("Flowkey → PDF")

        self.html_path = tk.StringVar()
        self.images_dir = tk.StringVar()
        self.measures_per_line = tk.DoubleVar(value=4.0)
        self.zoom = tk.DoubleVar(value=1.0)

        row = 0
        tk.Button(root, text="Select HTML/HTM…", command=self.pick_html).grid(row=row, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(root, textvariable=self.html_path, width=80).grid(row=row, column=1, padx=6, pady=6); row += 1

        tk.Label(root, text="Images folder:").grid(row=row, column=0, sticky="w", padx=6)
        tk.Entry(root, textvariable=self.images_dir, width=80).grid(row=row, column=1, padx=6, pady=6); row += 1

        tk.Button(root, text="1) Extract Images", command=self.extract_images).grid(row=row, column=0, sticky="w", padx=6, pady=6); row += 1

        tk.Label(root, text="Measures per line:").grid(row=row, column=0, sticky="w", padx=6)
        tk.Entry(root, textvariable=self.measures_per_line, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=6); row += 1

        tk.Label(root, text="Zoom (0.5–1.5 typical):").grid(row=row, column=0, sticky="w", padx=6)
        tk.Entry(root, textvariable=self.zoom, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=6); row += 1

        tk.Button(root, text="2) Build PDF", command=self.build_pdf).grid(row=row, column=0, sticky="w", padx=6, pady=12)
        self.status = tk.StringVar(value="Ready")
        tk.Label(root, textvariable=self.status, fg="gray").grid(row=row, column=1, sticky="w", padx=6)

    def pick_html(self):
        path = filedialog.askopenfilename(
            title="Select saved HTML",
            filetypes=[("HTML files", "*.html *.htm"), ("All files", "*.*")]
        )
        if not path: return
        self.html_path.set(path)
        self.images_dir.set(os.path.join(os.path.dirname(path), "images"))

    def extract_images(self):
        html = self.html_path.get().strip()
        out_dir = self.images_dir.get().strip()
        if not html:
            messagebox.showerror("Missing", "Please select an HTML/HTM file first."); return
        try:
            self.status.set("Extracting images…"); self.root.update_idletasks()
            n = extract_sheet_images_from_html(html, out_dir)
            if n == 0:
                messagebox.showwarning("No images", "No sheet images were found. Did you save after scrolling all pages?")
            else:
                messagebox.showinfo("Done", f"Saved {n} images to:\n{out_dir}")
        except Exception as e:
            messagebox.showerror("Error extracting", str(e))
        finally:
            self.status.set("Ready")

    def build_pdf(self):
        img_dir = self.images_dir.get().strip()
        if not img_dir or not os.path.isdir(img_dir):
            messagebox.showerror("Missing", "Images folder not found. Extract images first."); return
        out_pdf = filedialog.asksaveasfilename(
            title="Save PDF", defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile="score_stitched.pdf",
            initialdir=os.path.dirname(self.html_path.get()) if self.html_path.get() else None
        )
        if not out_pdf: return
        try:
            self.status.set("Building PDF…"); self.root.update_idletasks()
            mpl = float(self.measures_per_line.get()); zm = float(self.zoom.get())
            measures, pages, scale = make_pdf_from_measures(
                input_dir=img_dir,
                output_pdf=out_pdf,
                measures_per_line=mpl,
                zoom=zm,
                page="A4",
                orientation="portrait",
                dpi=300
            )
            messagebox.showinfo("Done", f"PDF saved:\n{out_pdf}\n\nMeasures: {measures}\nPages: {pages}\nScale: {scale:.3f}")
        except Exception as e:
            messagebox.showerror("Error building PDF", str(e))
        finally:
            self.status.set("Ready")

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
