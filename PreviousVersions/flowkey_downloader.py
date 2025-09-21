# pip install selenium webdriver-manager requests

import os, re, time, requests
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def download_sheet_images(
    url: str,
    image_selector: str = ".sheet-container .scrollable-sheet .sheet-image",
    out_dir: str = r"C:\Temp\flowkey_sheets",
    container_selector: str = ".scrollable-sheet",
    use_chrome_profile: str | None = None,     # e.g. r"C:\Users\<you>\AppData\Local\Google\Chrome\User Data"
    profile_dir_name: str = "Default",         # often "Default" or "Profile 1"
    headless: bool = False,                    # start False so you can log in; switch to True later
    scroll_pause: float = 0.7,
    max_stable_passes: int = 4,
    timeout_sec: int = 30,
):
    """
    Open the URL, scroll until all sheets load, collect each .sheet-image background-image URL, and download them.

    Tips:
      - For Flowkey, keep image_selector=".sheet-container .scrollable-sheet .sheet-image"
      - If you see 0 images, make sure you're logged in. Use headless=False or a chrome profile.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Start Chrome ---
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1400,900")

    # Reuse your Chrome profile (keeps cookies = logged-in)
    if use_chrome_profile:
        # user-data-dir must point to the parent of profiles, profile-directory picks the actual profile
        opts.add_argument(f'--user-data-dir={use_chrome_profile}')
        opts.add_argument(f'--profile-directory={profile_dir_name}')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.get(url)

    wait = WebDriverWait(driver, timeout_sec)

    # --- Wait for page to render *something*; Flowkey app may take a second
    # Try to wait for the container; if it doesn't appear, we still proceed and try the page body
    container = None
    try:
        container = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, container_selector)))
    except Exception:
        print(f"[warn] Could not find container '{container_selector}' within {timeout_sec}s. Will try the whole page.")

    # --- Scroll until stable (either container or body)
    def do_scroll():
        target = container if container is not None else driver.find_element(By.TAG_NAME, "body")
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", target)

    last_height, stable = -1, 0
    while stable < max_stable_passes:
        do_scroll()
        time.sleep(scroll_pause)
        if container is not None:
            height = driver.execute_script("return arguments[0].scrollHeight;", container)
        else:
            height = driver.execute_script("return document.body.scrollHeight;")
        if height == last_height:
            stable += 1
        else:
            stable = 0
            last_height = height

    # --- Wait for at least one image element to appear
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, image_selector)))
    except Exception:
        print(f"[error] No elements matched selector '{image_selector}'.")
        print("Possible reasons: not logged in, wrong selector, or content behind an interstitial.")
        driver.quit()
        return

    # --- Gather all elements & extract URLs (background-image or <img src>)
    elements = driver.find_elements(By.CSS_SELECTOR, image_selector)
    print(f"[info] Found {len(elements)} elements for selector '{image_selector}'")

    urls = []
    for el in elements:
        # Prefer CSS background-image (Flowkey uses this)
        bg = el.value_of_css_property("background-image")
        m = re.match(r'url\(["\']?(.*?)["\']?\)', bg or "")
        if m:
            urls.append(m.group(1))
            continue
        # Fallback: if it's actually an <img>, grab src
        try:
            img = el.find_element(By.TAG_NAME, "img")
            src = img.get_attribute("src")
            if src:
                urls.append(src)
        except Exception:
            pass

    # Deduplicate, preserve order
    seen = set()
    uniq_urls = []
    for u in urls:
        if u not in seen:
            uniq_urls.append(u)
            seen.add(u)

    print(f"[info] Extracted {len(uniq_urls)} unique image URLs")

    if not uniq_urls:
        print("[warn] No image URLs found. Are you logged in? Is the selector correct?")
        driver.quit()
        return

    # --- Download helper
    def guess_ext(u):
        path = urlparse(u).path.lower()
        m = re.search(r'\.(png|jpg|jpeg|webp|gif|svg)$', path)
        return '.' + (m.group(1) if m else 'jpg')

    # --- Download
    for i, u in enumerate(uniq_urls, 1):
        ext = guess_ext(u)
        name = f"sheet_{i:03d}{ext}"
        out_path = os.path.join(out_dir, name)
        print(f"[dl] {i}/{len(uniq_urls)} -> {name}")
        r = requests.get(u, timeout=30)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)

    driver.quit()
    print("[done] Saved", len(uniq_urls), "images to", out_dir)


# --- Example call (use this after saving the script as a .py file) ---
# download_sheet_images(
#     url="https://app.flowkey.com/player/26qrx2qukAtmyRQ4W",
#     image_selector=".sheet-container .scrollable-sheet .sheet-image",
#     out_dir=r"C:\Users\pmgar\OneDrive\Documents\FlowKey\Elvis",
#     use_chrome_profile=r"C:\Users\pmgar\AppData\Local\Google\Chrome\User Data",
#     profile_dir_name="Default",   # or "Profile 1", etc.
#     headless=False                # keep False the first time so you can log in if needed
# )

download_sheet_images(
    url="https://app.flowkey.com/player/26qrx2qukAtmyRQ4W",
    image_selector=".sheet-container .scrollable-sheet .sheet-image",
    out_dir=r"C:\Users\pmgar\OneDrive\Documents\FlowKey\Elvis",
    use_chrome_profile=r"C:\Users\pmgar\AppData\Local\Google\Chrome\User Data",  # this folder will be created
    profile_dir_name="Default",
    headless=False
)

#download_sheet_images(
#   url="https://app.flowkey.com/player/26qrx2qukAtmyRQ4W",
##    html_path=".sheet-container .scrollable-sheet .sheet-image",
##  out_dir=r"C:\Users\pmgar\OneDrive\Documents\FlowKey\Elvis"
#)

# #body > div:nth-child(1) > div > div > div.flowkey-player.css-z0hj4e > div.sheet-container > div.scrollable-sheet > div:nth-child(4)",
#

# Example usage:
# download_sheet_images(
#     url="https://www.flowkey.com/en/songs/example",
#     html_path="div.sheet-container div.scrollable-sheet div.sheet-image",
#     out_dir="C:/Users/YourName/Downloads/flowkey_sheets"
# )

#download_sheet_images(
 #   url="https://app.flowkey.com/player/26qrx2qukAtmyRQ4W",
  #  html_path="html body div div.css-v93k9i div.css-1ob46zo div.flowkey-player.css-z0hj4e div.sheet-container div.scrollable-sheet div.sheet-image",
   # out_dir=r"C:\Users\pmgar\OneDrive\Documents\FlowKey\Elvis"
#)
