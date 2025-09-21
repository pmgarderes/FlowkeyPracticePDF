# flowkey2pdf_gui_debug.py
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
from PIL import Image, ImageOps, ImageDraw
from io import BytesIO


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
        with open(path, "wb") as f: f.write(r.content)

    def save_file(u: str, path: str):
        parsed = urlparse(u)
        src_path = url2pathname(parsed.path)
        if not os.path.isabs(src_path):
            src_path = os.path.join(html_dir, src_path)
        shutil.copyfile(src_path, path)

    def save_data(u: str, path: str):
        head, b64 = u.split(",", 1)
        with open(path, "wb") as f: f.write(base64.b64decode(b64))

    count = 0
    # --- minimal change: normalize every saved file to RGB PNG (no palettes/alpha/ICC) ---
    def _save_png_normalized_from_bytes(raw_bytes: bytes, out_path: str):
        im = Image.open(BytesIO(raw_bytes))
        if im.mode in ("RGBA", "LA") or ("A" in im.getbands()):
            bg = Image.new("RGB", im.size, "white")
            alpha = im.split()[-1] if "A" in im.getbands() else None
            bg.paste(im.convert("RGB"), (0, 0), alpha)
            im = bg
        else:
            im = im.convert("RGB")
        im.save(out_path, format="PNG", optimize=True)

    count = 0
    for i, (_, top_val, raw_u, final) in enumerate(resolved, 1):
        out_name = f"sheet_{i:03d}.png"  # force PNG output
        out_path = os.path.join(out_dir, out_name)
        try:
            if final.startswith("data:"):
                head, b64 = final.split(",", 1)
                raw = base64.b64decode(b64)
                _save_png_normalized_from_bytes(raw, out_path)
            else:
                scheme = urlparse(final).scheme.lower()
                if scheme in ("http", "https"):
                    r = requests.get(final, timeout=30)
                    r.raise_for_status()
                    _save_png_normalized_from_bytes(r.content, out_path)
                elif scheme == "file":
                    parsed = urlparse(final)
                    src_path = url2pathname(parsed.path)
                    if not os.path.isabs(src_path):
                        src_path = os.path.join(html_dir, src_path)
                    with open(src_path, "rb") as f:
                        _save_png_normalized_from_bytes(f.read(), out_path)
                else:
                    with open(os.path.join(html_dir, raw_u), "rb") as f:
                        _save_png_normalized_from_bytes(f.read(), out_path)
            count += 1
        except Exception as e:
            print("[error]", final, "->", e)

# -------------------- images -> PDF --------------------

def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))

def page_pixels(page: str = "A4", orientation: str = "portrait", dpi: int = 300):
    sizes_mm = {"A4": (210, 297), "Letter": (215.9, 279.4)}
    w_mm, h_mm = sizes_mm[page]
    if orientation.lower() == "landscape":
        w_mm, h_mm = h_mm, w_mm
    return mm_to_px(w_mm, dpi), mm_to_px(h_mm, dpi)

def load_and_stitch(input_dir: str):
    names = sorted(n for n in os.listdir(input_dir)
                   if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    if not names:
        raise FileNotFoundError(f"No images found in {input_dir}")

    imgs, widths = [], []
    for n in names:
        im = Image.open(os.path.join(input_dir, n))
        if im.mode in ("RGBA","LA") or ("A" in im.getbands()):
            bg = Image.new("RGB", im.size, "white")
            alpha = im.split()[-1] if "A" in im.getbands() else None
            bg.paste(im.convert("RGB"), (0,0), alpha)
            im = bg
        else:
            im = im.convert("RGB")
        THR = 10  # tweak 180–210 if needed
        im = im.convert("L").point(lambda p: 0 if p < THR else 255, "L").convert("RGB")
        # im = ImageOps.grayscale(im).convert("RGB")  # ← 1 line: kills green cast
        imgs.append(im)
        widths.append(im.width)

    H = imgs[0].height
    W = int(sum(widths))
    stitched = Image.new("RGB", (W, H), "white")
    x = 0
    for im in imgs:
        stitched.paste(im, (x, 0)); x += im.width
    return stitched

def detect_measures(stitched: Image.Image, thr: int | None = 200,
                    min_height_ratio: float = 0.50,
                    min_cont_ratio: float = 0.50):
    # Use normalized grayscale for robust thresholding
    arr = prep_gray(stitched, low_pct=1, high_pct=99)   # << NEW
    H, W = arr.shape

    # threshold: if None -> Otsu on the normalized image
    if thr is None:
        # Otsu on arr
        hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
        total = arr.size
        csum = hist.cumsum()
        msum = (hist * np.arange(256)).cumsum()
        denom = csum * (total - csum)
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma_b2 = (msum*total - csum*msum[-1])**2 / np.where(denom == 0, 1, denom)
        thr = int(np.nanargmax(sigma_b2))

    bw = (arr < thr)


    rows_any = bw.any(axis=1)
    if rows_any.any():
        top = int(np.argmax(rows_any))
        bottom = H - 1 - int(np.argmax(rows_any[::-1]))
    else:
        top, bottom = 0, H - 1
    content_h = max(1, bottom - top + 1)
    band = bw[top:bottom+1, :]

    col_frac = band.mean(axis=0)

    max_run = np.zeros(W, dtype=int)
    for xi in range(W):
        col = band[:, xi]
        longest = cur = 0
        for v in col:
            if v: cur += 1; longest = max(longest, cur)
            else: cur = 0
        max_run[xi] = longest
    cont_frac = max_run / float(content_h)

    mask = (col_frac >= min_height_ratio) & (cont_frac >= min_cont_ratio)

    # contiguous mask runs -> bar spans
    runs = []
    in_run = False; start = 0
    for xi, m in enumerate(mask):
        if m and not in_run: in_run = True; start = xi
        elif not m and in_run: in_run = False; runs.append((start, xi-1))
    if in_run: runs.append((start, W-1))

    # measures without gaps: barline goes with RIGHT measure
    measures = []
    prev = 0
    for a, b in runs:
        if a > prev: measures.append((prev, a-1))
        prev = a
    if prev < W: measures.append((prev, W-1))
    measures = [(x0, x1) for (x0, x1) in measures if x1 >= x0]

    info = {
        "thr": thr, "H": H, "W": W,
        "content_top": top, "content_bottom": bottom,
        "num_bars": len(runs), "num_measures": len(measures),
        "col_frac_max": float(col_frac.max()) if col_frac.size else 0.0,
        "cont_frac_max": float(cont_frac.max()) if cont_frac.size else 0.0,
        "runs": runs,
        "measures": measures,
        "binary_band": band,      # for optional debug image
    }
    return measures, info

def make_pdf_from_measures(
    input_dir: str,
    output_pdf: str,
    measures_per_line: float,
    zoom: float = 1.0,
    page: str = "A4",
    orientation: str = "portrait",
    dpi: int = 300,
    margin_mm: float = 10.0,
    thr: int = 200,
    min_height_ratio: float = 0.50,
    min_cont_ratio: float = 0.50,
    debug_dir: str | None = None,
):
    stitched = load_and_stitch(input_dir)
    measures, info = detect_measures(stitched, thr, min_height_ratio, min_cont_ratio)

    if debug_dir:
        save_debug_overlays(stitched, info, debug_dir)

    if not measures:
        raise RuntimeError("No measures detected with current parameters. Try lowering Min height/continuous %, or adjust threshold.")

    page_w, page_h = page_pixels(page, orientation, dpi)
    margin_px = mm_to_px(margin_mm, dpi)
    content_w = page_w - 2*margin_px
    content_h_px = page_h - 2*margin_px

    # scale whole image once; then crop (no seams)
    H = stitched.height
    W = stitched.width
    avg_w = float(np.mean([(x1-x0+1) for (x0,x1) in measures]))
    base_scale = (content_w) / (avg_w * measures_per_line)
    scale = max(0.05, min(5.0, base_scale * zoom))

    W_scaled = int(round(W * scale))
    H_scaled = int(round(H * scale))
    scaled = stitched.resize((W_scaled, H_scaled), Image.BICUBIC)
    #scaled = stitched.resize((W_scaled, H_scaled), Image.BICUBIC).convert("L").convert("RGB")

    # scaled measure positions
    scaled_measures = []
    cursor = 0
    for (x0, x1) in measures:
        w = (x1 - x0 + 1)
        w_s = max(1, int(round(w * scale)))
        scaled_measures.append((cursor, cursor + w_s))
        cursor += w_s
    if scaled_measures:
        s0, s1 = scaled_measures[-1]
        scaled_measures[-1] = (s0, W_scaled)

    # paginate
    pages = []
    page_img = Image.new("RGB", (page_w, page_h), "white")
    x = margin_px; y = margin_px; row_h = 0
    row_spacing = mm_to_px(3.0, dpi)  # only vertical spacing
    col_spacing = 0                    # no horizontal gaps

    def new_page():
        nonlocal page_img, x, y, row_h
        pages.append(page_img)
        page_img = Image.new("RGB", (page_w, page_h), "white")
        x = margin_px; y = margin_px; row_h = 0

    for (sx0, sx1) in scaled_measures:
        crop = scaled.crop((sx0, 0, sx1, H_scaled))
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
    return len(measures), len(pages), scale, info

def prep_gray(stitched_img, low_pct=1, high_pct=99):
    """
    Convert to grayscale and stretch contrast between percentiles
    (e.g., 1st and 99th) so black/white separation is consistent.
    Returns uint8 ndarray (H,W).
    """
    g = ImageOps.grayscale(stitched_img)
    arr = np.asarray(g, dtype=np.uint8)

    lo, hi = np.percentile(arr, (low_pct, high_pct))
    if hi <= lo:            # fallback if image already maxed
        return arr
    arr2 = np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return arr2
# -------------------- debug overlays --------------------

def save_debug_overlays(stitched: Image.Image, info: dict, debug_dir: str):
    os.makedirs(debug_dir, exist_ok=True)
    stitched.save(os.path.join(debug_dir, "stitched.png"))

    g = ImageOps.grayscale(stitched)
    g.save(os.path.join(debug_dir, "stitched_gray.png"))

    thr = info["thr"]
    arr = np.asarray(g, dtype=np.uint8)
    bw = (arr < thr).astype(np.uint8) * 255
    Image.fromarray(bw, mode="L").save(os.path.join(debug_dir, f"binary_thr{thr}.png"))

    # overlay bars (red) and measure boundaries (green)
    overlay = stitched.copy()
    draw = ImageDraw.Draw(overlay)
    H = stitched.height

    # bar centers (red)
    for (a,b) in info["runs"]:
        x = (a + b)//2
        draw.line([(x, 0), (x, H)], fill=(255, 0, 0), width=2)

    # measure boundaries (green)
    for (x0, x1) in info["measures"]:
        draw.line([(x0, 0), (x0, H)], fill=(0, 180, 0), width=1)
    # last edge
    if info["measures"]:
        draw.line([(info["measures"][-1][1], 0), (info["measures"][-1][1], H)], fill=(0, 180, 0), width=1)

    overlay.save(os.path.join(debug_dir, f"overlay_thr{thr}_h{int(100*0.5)}_c{int(100*0.5)}.png"))

    # small report
    with open(os.path.join(debug_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Detection report\n"
            f"W x H = {info['W']} x {info['H']}\n"
            f"content band: top={info['content_top']} bottom={info['content_bottom']}\n"
            f"thr={thr}\n"
            f"bars={info['num_bars']} measures={info['num_measures']}\n"
        )

# -------------------- GUI --------------------

class App:
    def __init__(self, root):
        self.root = root
        root.title("Flowkey → PDF (with Debug)")

        self.html_path = tk.StringVar()
        self.images_dir = tk.StringVar()
        self.measures_per_line = tk.DoubleVar(value=4.0)
        self.zoom = tk.DoubleVar(value=1.0)

        # Advanced / debug vars
        self.show_adv = tk.BooleanVar(value=False)
        self.debug_on = tk.BooleanVar(value=True)
        self.thr_str = tk.StringVar(value="10")  # blank => Otsu, else int
        self.min_height_pct = tk.IntVar(value=50) # 50%
        self.min_cont_pct = tk.IntVar(value=50)   # 50%

        # Row 0-2: file pick + images dir
        row = 0
        tk.Button(root, text="Select HTML/HTM…", command=self.pick_html).grid(row=row, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(root, textvariable=self.html_path, width=80).grid(row=row, column=1, padx=6, pady=6); row += 1

        tk.Label(root, text="Images folder:").grid(row=row, column=0, sticky="w", padx=6)
        tk.Entry(root, textvariable=self.images_dir, width=80).grid(row=row, column=1, padx=6, pady=6); row += 1

        tk.Button(root, text="1) Extract Images", command=self.extract_images).grid(row=row, column=0, sticky="w", padx=6, pady=6)
        tk.Checkbutton(root, text="Show advanced/debug", variable=self.show_adv, command=self.toggle_adv).grid(row=row, column=1, sticky="w"); row += 1

        # Simple controls
        tk.Label(root, text="Measures per line:").grid(row=row, column=0, sticky="w", padx=6)
        tk.Entry(root, textvariable=self.measures_per_line, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=6); row += 1
        tk.Label(root, text="Zoom (0.5–1.5 typical):").grid(row=row, column=0, sticky="w", padx=6)
        tk.Entry(root, textvariable=self.zoom, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=6); row += 1

        tk.Button(root, text="2) Build PDF", command=self.build_pdf).grid(row=row, column=0, sticky="w", padx=6, pady=12)

        self.status = tk.StringVar(value="Ready")
        tk.Label(root, textvariable=self.status, fg="gray").grid(row=row, column=1, sticky="w", padx=6); row += 1

        # Advanced frame (hidden by default)
        self.adv = tk.Frame(root, bd=1, relief="groove")
        rr = 0
        tk.Label(self.adv, text="Detection threshold (blank = Otsu):").grid(row=rr, column=0, sticky="w", padx=6, pady=3)
        tk.Entry(self.adv, textvariable=self.thr_str, width=8).grid(row=rr, column=1, sticky="w"); rr += 1

        tk.Label(self.adv, text="Min height %:").grid(row=rr, column=0, sticky="w", padx=6)
        tk.Spinbox(self.adv, from_=30, to=90, textvariable=self.min_height_pct, width=5).grid(row=rr, column=1, sticky="w"); rr += 1

        tk.Label(self.adv, text="Min continuous %:").grid(row=rr, column=0, sticky="w", padx=6)
        tk.Spinbox(self.adv, from_=30, to=90, textvariable=self.min_cont_pct, width=5).grid(row=rr, column=1, sticky="w"); rr += 1

        tk.Checkbutton(self.adv, text="Save debug overlays", variable=self.debug_on).grid(row=rr, column=0, sticky="w", padx=6); rr += 1

        tk.Button(self.adv, text="Probe detection (save overlays)", command=self.probe_detection).grid(row=rr, column=0, sticky="w", padx=6, pady=6)

    def toggle_adv(self):
        if self.show_adv.get():
            self.adv.grid(row=8, column=0, columnspan=2, sticky="we", padx=6, pady=6)
        else:
            self.adv.grid_forget()

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

    def _parse_thr(self):
        s = self.thr_str.get().strip()
        if not s: return None
        try:
            v = int(s); return max(0, min(255, v))
        except: return None

    def _debug_dir(self):
        html = self.html_path.get().strip()
        base = os.path.dirname(html) if html else os.getcwd()
        dd = os.path.join(base, "debug")
        os.makedirs(dd, exist_ok=True)
        return dd

    def probe_detection(self):
        img_dir = self.images_dir.get().strip()
        if not img_dir or not os.path.isdir(img_dir):
            messagebox.showerror("Missing", "Images folder not found. Extract images first."); return
        try:
            self.status.set("Probing…"); self.root.update_idletasks()
            stitched = load_and_stitch(img_dir)
            thr = self._parse_thr()
            if thr is None:
                # quick Otsu
                g = ImageOps.grayscale(stitched)
                arr = np.asarray(g, dtype=np.uint8)
                # Otsu
                hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
                total = arr.size
                csum = hist.cumsum()
                msum = (hist * np.arange(256)).cumsum()
                denom = csum * (total - csum)
                with np.errstate(divide='ignore', invalid='ignore'):
                    sigma_b2 = (msum*total - csum*msum[-1])**2 / np.where(denom == 0, 1, denom)
                thr = int(np.nanargmax(sigma_b2))

            measures, info = detect_measures(
                stitched,
                thr=thr,
                min_height_ratio=self.min_height_pct.get()/100.0,
                min_cont_ratio=self.min_cont_pct.get()/100.0
            )
            dd = self._debug_dir()
            save_debug_overlays(stitched, info, dd)
            os.startfile(dd)  # open folder on Windows
            messagebox.showinfo("Probe result",
                                f"Bars: {info['num_bars']}\nMeasures: {info['num_measures']}\n"
                                f"thr={thr}, min height={self.min_height_pct.get()}%, min cont={self.min_cont_pct.get()}%\n"
                                f"Debug saved to:\n{dd}")
        except Exception as e:
            messagebox.showerror("Probe error", str(e))
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
            thr = self._parse_thr()
            # if blank, default to 200 as before
            if thr is None: thr = 200
            debug_dir = self._debug_dir() if self.debug_on.get() else None

            measures, pages, scale, info = make_pdf_from_measures(
                input_dir=img_dir,
                output_pdf=out_pdf,
                measures_per_line=mpl,
                zoom=zm,
                page="A4",
                orientation="portrait",
                dpi=300,
                thr=thr,
                min_height_ratio=self.min_height_pct.get()/100.0,
                min_cont_ratio=self.min_cont_pct.get()/100.0,
                debug_dir=debug_dir
            )
            if debug_dir:
                os.startfile(debug_dir)
            messagebox.showinfo("Done", f"PDF saved:\n{out_pdf}\n\n"
                                        f"Measures: {measures}\nPages: {pages}\nScale: {scale:.3f}\n"
                                        f"thr={thr}, min height={self.min_height_pct.get()}%, min cont={self.min_cont_pct.get()}%")
        except Exception as e:
            messagebox.showerror("Error building PDF", str(e))
        finally:
            self.status.set("Ready")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
