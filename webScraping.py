from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
import urllib.request
import time


#lists to store scraped data
authors=[]
dates=[]
statements=[]
sources=[]
targets=[]



def scrape_website(page_number):
    page_num = str(page_number)
    url = 'https://www.politifact.com/factchecks/list/?page=' + page_num
    webpage = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(webpage.text, 'html.parser')


    statement_footer = soup.find_all('footer', class_='m-statement__footer')
    statement_quote = soup.find_all('div', class_='m-statement__quote')
    statement_meta = soup.find_all('div', class_='m-statement__meta')
    target = soup.find_all('div', class_='m-statement__meter')


    # Author + Date
    for i in statement_footer:
        text = i.text.strip().split()

        # defensive check
        if len(text) >= 7:
            full_name = text[1] + ' ' + text[2]
            date = text[4] + ' ' + text[5] + ' ' + text[6]

            authors.append(full_name)
            dates.append(date)


    # Statement

    for i in statement_quote:
        a_tag = i.find('a')
        if a_tag:
            statements.append(a_tag.text.strip())


    # Source

    for i in statement_meta:
        a_tag = i.find('a')
        if a_tag:
            sources.append(a_tag.text.strip())


    # Verdict / Target

    for i in target:
        img = i.find('img')
        if img and img.get('alt'):
            targets.append(img.get('alt').strip())

