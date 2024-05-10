import requests


def download_file(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)


# Example usage
onedrive_link = 'https://anu365-my.sharepoint.com/:u:/r/personal/u1093778_anu_edu_au/Documents/g-drift/1d_prem.h5?csf=1&web=1&e=y2TZvQ'
save_path = ''
download_file(onedrive_link, save_path)
