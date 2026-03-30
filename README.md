# FlowkeyPracticePDF

A tiny graphical user interface that turns a locally saved Flowkey page into a clean, paginated PDF of the sheet music.

---
## ⭐ Stay Updated & Like 
if you find it usefull, This helps me know people are interested, and I’ll post updates or improvements there. 


⚠️ **Disclaimer**  
This project is provided for educational and personal use only.  
- You may use this tool to generate sheet music **only for your own practice**, in accordance with your Flowkey subscription.  
- **Do not** use this tool to publish, distribute, or share copyrighted music scores obtained from Flowkey or any third-party service.  
- Any misuse of this tool may violate copyright law and Flowkey’s terms of service.  
- See the [LICENSE](LICENSE) file for full details.

---

## Typical workflow (~1 min) - see details below - 
1. Save a HTML file from your internet browser.  
2. run the code to start the graphical user interface
3. select parameters for the final layout ( measures per line, zoom..)   
4. Click "save PDF" and print the music sheet !



---

## What this code does
1. Reads your **saved HTML/HTM** file.  
2. Extracts all sheet images into an `images/` folder beside the HTML.  
3. Detects **barlines** and wraps lines **only at measure boundaries**.  
4. Builds an **A4 PDF** with your chosen layout.


---

## 🚀 Installation
Clone this repository:

```bash
git clone https://github.com/pmgarderes/FlowkeyPracticePDF.git
cd FlowkeyPracticePDF
```

Install dependencies (Python 3.8+ recommended):

```bash
pip install -r requirements.txt
```

---

## 📖 Usage


## I - Save the HTML correctly (important)
1. Open the song in your browser.
2. (might be needed- see later) **Scroll through all pages** so every sheet image loads.
3. Save the page locally (e.g., right-click-> “Save page as…” → **Webpage, Complete**).
   - You’ll get `SongName.htm(l)` in a folder you choose.

---

## II - Run the app

```bash
python flowkey2pdf_gui_debug.py
```

---

## III - Use the GUI to save the sheet music
1. **Select HTML/HTM…**  
   Pick the `.html`/`.htm` you saved. The app auto-fills the `images/` folder beside it.

2. **1) Extract Images**  
   - Downloads/copies every page of the sheet into `<html_folder>\images\sheet_###.png`.  
   - If it says “No images”, make sure you scrolled the whole song before saving.

3. **Set layout**
   - **Measures per line**: e.g., `4.0` (average target).  
   - **Zoom**: `< 1.0` = smaller (more content per page), `> 1.0` = larger.  
   - **Hard threshold (binarize tiles)** *(optional)*: check this **only if some images appear with a colored or tinted background (e.g., green hue)**.  
     When enabled, it converts each tile to pure black & white before stitching. Normally this should stay **unchecked**, since it can slightly thin out staff lines.

4. **(Optional) Show advanced/debug**
   - **Threshold**: leave blank to auto-detect (Otsu on normalized grayscale), or type a number (e.g., `200`).  
   - **Min height %**: how much of the staff height a vertical line must cover to count as a barline (default 50).  
   - **Min continuous %**: how uninterrupted that vertical line must be (default 50).  
   - **Probe detection** saves overlays (`debug/` folder) showing detected bars (red) and measure boundaries (green).  

   **Note:** If you notice missing lines or other problems, please enable debugging, check “Save debug overlays”, and send me the resulting images from the `debug/` folder together with the parameters you used — this will help identify the issue.

5. **Build PDF**  
   - Choose the output filename.  
   - The app writes a clean **A4** PDF with no gaps between measures.

---

## Typical workflow (30 seconds)
1. **Select HTML → Extract Images**  
2. **Measures per line = `4.0`, Zoom = `0.9`**  
3. **(Optional) Probe detection** → check `debug/overlay_*.png`  
4. **Build PDF**

---

## Output layout
```
YourSongFolder/
├─ YourSong.htm(l)
├─ images/
│  ├─ sheet_001.png
│  ├─ sheet_002.png
│  └─ ...
├─ debug/                (only if you used debugging)
│  ├─ stitched.png
│  ├─ stitched_gray.png
│  ├─ gray_norm.png
│  ├─ binary_norm_thrXXX.png
│  └─ overlay_thr...png
└─ YourPDF.pdf
```


---


## 📜 License
This project is licensed under a **Personal, Non-Commercial Use License**.  
See [LICENSE](LICENSE) for details.


## Tips & troubleshooting
- **Green-tinted pages?**  
  The app converts images to **grayscale** internally before layout—this eliminates palette/ICC issues. If you still see tint, ensure you’re running the latest script and not an older version still in memory.

- **“No images found”**  
  - Scroll the whole piece before saving the HTML.  
  - Some songs may use different selectors; default is `.sheet-container .scrollable-sheet .sheet-image`.

- **“No measures detected” or wrong splits**  
  - Click **Show advanced/debug → Probe detection**.  
  - Try a **Threshold** of `180–210`, or leave blank for auto.  
  - If barlines don’t span both staves: set **Min height %** to `45`, **Min continuous %** to `45`.  
  - Check `debug/overlay_*.png` to see what the detector saw.

- **Paths / permissions**  
  - If saving to OneDrive causes “Access denied”, choose another folder.  
  - The GUI uses a file picker, so you don’t need to type paths.

- **Speed**  
  - Large songs: the first stitch/scale is the slowest step. It only happens once per “Build PDF”.

