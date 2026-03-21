import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# ── Setup headless Chrome ─────────────────────────────────
def create_driver():
    options = Options()
    options.add_argument("--headless")          # no visible window
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,800")
    options.add_argument("--disable-gpu")
    
    # automatically downloads correct Chrome driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# ── Take a screenshot of one URL ─────────────────────────
def screenshot_url(url, save_path, timeout=10):
    driver = None
    try:
        driver = create_driver()
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        time.sleep(2)  # wait for page to load
        driver.save_screenshot(save_path)
        print(f"Saved: {save_path}")
        return True
    except Exception as e:
        print(f"Failed: {url} — {e}")
        return False
    finally:
        if driver:
            driver.quit()

# ── Test on one URL first ─────────────────────────────────
os.makedirs('screenshots/test', exist_ok=True)

success = screenshot_url(
    'https://www.google.com',
    'screenshots/test/google.png'
)

if success:
    print("Selenium is working correctly!")
else:
    print("Something went wrong — check the error above")