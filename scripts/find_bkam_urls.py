import requests
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

resp = requests.get("https://www.bkam.ma", verify=False)
soup = BeautifulSoup(resp.text, 'html.parser')

print("Found Publications URLs:")
for a in soup.find_all('a', href=True):
    href = a['href']
    if 'publication' in href.lower() or 'rapport' in href.lower():
        if href.startswith('/'):
            href = 'https://www.bkam.ma' + href
        print(href)
