import requests
import pathlib
import sys


def download_database():
    url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    save_path = f'{pathlib.Path().absolute()}/cats_and_dogs_filtered.zip'
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        total_length = r.headers.get('content-length')

        if total_length is None:  # no content length header
            fd.write(r.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in r.iter_content(chunk_size=4096):
                dl += len(data)
                fd.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)))
                sys.stdout.flush()
    return (pathlib.Path().absolute(), save_path,)

# wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O /tmp/cats_and_dogs_filtered.zip
