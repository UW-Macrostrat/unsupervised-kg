from selenium.webdriver.chrome.options import Options
import time
from selenium import webdriver
from chromedriver_py import binary_path
import os

def get_chrome_driver():
    svc = webdriver.ChromeService(executable_path=binary_path)
    driver = webdriver.Chrome(service=svc)
    return driver

def visualize_web_pages():
    samples_dir = "samples"
    driver = get_chrome_driver()

    for file_name in os.listdir(samples_dir):
        if file_name[0] == '.' or "html" not in file_name:
            continue
        
        # Load the html file
        html_path = os.path.join(samples_dir, file_name)
        complete_path = os.path.abspath(html_path)
        driver.get("file://" + complete_path)
        time.sleep(5)

        # Take a screen shot and take an image
        save_path = os.path.join(samples_dir, file_name.replace("html", "png"))
        driver.save_screenshot(save_path)
    
    driver.quit()

if __name__ == "__main__":
    visualize_web_pages()