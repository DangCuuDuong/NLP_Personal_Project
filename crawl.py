import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

def crawl_amazon_product_reviews(product_url, output_file='amazon_reviews.csv'):
    # ==== Cấu hình trình duyệt ====
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Bật nếu muốn chạy nền
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        # ==== Mở trang sản phẩm ====
        driver.get(product_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(2)

        # ==== Cuộn trang để load đánh giá ====
        for _ in range(20):
            driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(1)

        # ==== Mở file CSV ====
        with open(output_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Rating", "Comment"])

            # ==== Lấy đánh giá ====
            reviews = driver.find_elements(By.CSS_SELECTOR, '[data-hook="review"]')
            print(f"🔍 Tìm thấy {len(reviews)} đánh giá")

            for r in reviews:
                try:
                    WebDriverWait(r, 5).until(EC.presence_of_element_located((By.CLASS_NAME, "a-icon-alt")))
                    rating_tag = r.find_element(By.CLASS_NAME, "a-icon-alt")
                    rating = rating_tag.text.strip() or rating_tag.get_attribute("innerHTML").strip()
                    rating = int(float(rating.split(" ")[0]))

                    comment = r.find_element(By.CSS_SELECTOR, '[data-hook="review-body"] span').text.strip()

                    print("⭐", rating)
                    print("📝", comment)
                    print("------")

                    writer.writerow([rating, comment])
                except Exception as e:
                    print("⚠️ Lỗi khi lấy review:", e)

        print(f"✅ Đã lưu vào {output_file}")
    finally:
        driver.quit()

# Ví dụ sử dụng
# crawl_amazon_product_reviews("https://www.amazon.com/dp/ASIN")
