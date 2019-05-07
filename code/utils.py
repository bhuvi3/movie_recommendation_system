import os
from io import BytesIO

import pandas as pd
import requests
from PIL import Image


class ImageFetcher:
    """
    Pulls images from url present in a csv file and dumps them into a folder.
    
    """
    
    def __init__(self, data_file_path, url_column_name, output_folder_path, output_postfix=None, 
                 url_prefix=None, max_attempts=5):
        """
        Just your standard class initialization. 🤷
        
        Input:
        -------
        data_file_path (str): Path to the csv file which contains the URLs. 
        url_column_name (str): Name of the column in the csv file which contains the URLs.
        output_folder_path (str): Path to the output folder where the images need to be dumped.
        output_postfix (str): Postfix string appended to the output file.
        url_prefix (str): Base URL string.
        max_attempts(int): Maximum number of attempts to fetch the file.
        
        """
        self._data_file_path = data_file_path
        self._data_file_path = data_file_path 
        self._url_column_name = url_column_name 
        self._output_folder_path = output_folder_path   
        self._url_prefix = url_prefix
        self._max_attempts = max_attempts
        
        if not output_postfix:
            self._output_postfix = ''
        else:
            self._output_postfix = '_' + output_postfix 

    def _download_image(self, url, image_file_path, attempt=0):
        """
        Downloads the image and dumps to a file.

        Input:
        -------
        url (str): URL to the image which needs to be downloaded.
        image_file_path (str): Path to the output file.

        """
        r = requests.get(url)
        if r.status_code != requests.codes.ok:
            if attempt < self._max_attempts:
                self._download_image(url, image_file_path, attempt + 1)
                return
            else:
                assert False, f'Status code error while downloading %s: %s.'%(url, r.status_code)

        with Image.open(BytesIO(r.content)) as im:
            im.save(image_file_path)
            
    def _get_urls(self):
        """
        Reads the csv file and returns a list of all the URLs after prepending base path.
        
        Output:
        --------
        urls (list): List of URLs.
        
        """
        data_file = pd.read_csv(self._data_file_path)
        url_column = data_file[self._url_column_name]
        urls = url_column.values.tolist()
        
        if self._url_prefix:
            urls = [self._url_prefix + url for url in urls]
        
        return urls
    
    def _url_to_filepath(self, url):
        """
        Extracts filename from URL and returns the output path after appending the postfix string.
        
        Input:
        -------
        url (str): URL of the image.
        
        Output:
        --------
        filepath (str): Output path for the image.
        
        """
        filepath, ext = os.path.splitext(url)
        filename = os.path.split(filepath)[1]
        
        filepath = os.path.join(self._output_folder_path, filename + self._output_postfix + ext)
        
        return filepath
    
    def fetch_images(self):
        """
        Download images and saves them to the folder.
        
        """
        urls = self._get_urls()
        
        for url in urls:
            output_filepath = self._url_to_filepath(url)
            self._download_image(url, output_filepath)
