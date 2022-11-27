import gdown
import zipfile
import os


def download_images():

    if not os.path.exists(os.path.join(os.getcwd(), 'test_images')):
        url = 'https://drive.google.com/u/0/uc?id=1bv7yCj_mS8YkXFDhzJgXTIXNSpNODncL&export=download'
        output = 'test_images.zip'
        gdown.download(url, output, quiet=False)


        with zipfile.ZipFile('test_images.zip', 'r') as zip_ref:
            zip_ref.extractall('./')

def download_checkpoints():
    if not os.path.exists(os.path.join(os.getcwd(), 'trained_models')):
        url = 'https://drive.google.com/uc?id=1fWHwYCCnLOuuZNKeByCMga4m-IIXwjPD&export=download'
        output = 'trained_models.zip'
        gdown.download(url, output, quiet=False)

        with zipfile.ZipFile('trained_models.zip', 'r') as zip_ref:
            zip_ref.extractall('./')