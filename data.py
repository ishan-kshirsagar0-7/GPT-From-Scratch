import requests
from bs4 import BeautifulSoup
import time

# Function to scrape script and append to file
def scrape_script(url, file):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        script_container = soup.find('div', class_='scrolling-script-container')
        if script_container:
            script_text = script_container.get_text(separator='\n')
            with open(file, 'a', encoding='utf-8') as f:
                f.write(script_text + '\n\n')
            return True
    return False

# Base URL
base_url = 'https://www.springfieldspringfield.co.uk/view_episode_scripts.php?tv-show={}&episode=s{:02}e{:02}'

# TV show name
tv_show = 'modern-family'  # Replace with the desired TV show name
output_file = 'scripts.txt'

season = 8
while True:
    episode = 1
    while True:
        url = base_url.format(tv_show, season, episode)
        success = scrape_script(url, output_file)
        if not success:
            break
        print(f'Successfully scraped Season {season}, Episode {episode}')
        episode += 1
        time.sleep(1)  # Be polite and avoid overloading the server
    if episode == 1:  # If no episodes were found in the season, break the loop
        break
    season += 1

print('Finished scraping all available episodes.')