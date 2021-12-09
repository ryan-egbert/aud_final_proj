import urllib.request
from bs4 import BeautifulSoup
import pickle as pck
from math import ceil
import time
import numpy as np

def parse_url(url):
    try:
        with urllib.request.urlopen(url) as response:
            html = response.read()
    except:
        return f"Error with url: {url}"

    text = str(html)

    soup = BeautifulSoup(text, 'html.parser')

    category = soup.find('li', class_="crumb category").get_text().replace("\\n","").strip()

    header = soup.find('h1', class_="postingtitle")
    header = header.get_text().replace("\\n","").lstrip()

    body = soup.find('section', id="postingbody")
    body = body.get_text().replace("\\n", " ").replace("QR Code Link to This Post", "").strip()

    post_id = soup.find('div', class_="postinginfos").findAll('p')[0].get_text().replace("post id: ", "")

    return (category, header + body, post_id)

def get_text_from_urls(search_urls):
    Craigslinks=[]
    for url in search_urls:
        html = urllib.request.urlopen(url).read().decode('utf-8') #loading each search page
        #we should pause to let the results to load
        # time.sleep(np.random.uniform(10,15))
        #sleep time follows uniform distribution [100,150] with mean is 125

        index = html.find('<div class="open-map-view-button"') #going closer to the results table
        html2 = html[index:]

        while html2.find('<a href="https') != -1:
            #We are running a while loop because we want to get everything in the page

            index = html2.find('<div class="result-info">')
            remaining = html2[index:]

            start = remaining.find("<a href=")
            end = remaining.find(".html")
            Craigslinks.append(remaining[start+9:end+5])

            remaining = remaining[end:]
            html2 = remaining

    Craigslinks_set = set(Craigslinks)
    data = []

    for url in Craigslinks_set:
        data.append(parse_url(url))

    return data

def get_total_count(url):
    with urllib.request.urlopen(url) as response:
        html = response.read()
        text = str(html)
        soup = BeautifulSoup(text, 'html.parser')
        total = int(soup.find('span', class_="totalcount").get_text())

    return total

def main():
    search_urls=[]

    resume = "https://chicago.craigslist.org/d/resumes/search/rrr"
    job = "https://chicago.craigslist.org/d/jobs/search/jjj"
    counter = 120

    r_total = get_total_count(resume)
    j_total = get_total_count(job)

    for url, total in [(resume, r_total), (job, j_total)]:
        search_urls.append(url)
        for i in range(1,ceil(total/120)):
            search_urls.append(url + "?s={}".format(counter*i))

    data = get_text_from_urls(search_urls)

    print(len(data))
    
    with open("../pck/cl_data.pck", 'wb') as out:
        pck.dump(data, out)

main()