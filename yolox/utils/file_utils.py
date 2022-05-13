import os
import zipfile


def compress_file(zipfilename, dirname):
    if os.path.isfile(dirname):
        with zipfile.ZipFile(zipfilename, 'w') as z:
            z.write(dirname)
    else:
        with zipfile.ZipFile(zipfilename, 'w') as z:
            for root, _, files in os.walk(dirname):
                for single_file in files:
                    if single_file != zipfilename:
                        filepath = os.path.join(root, single_file)
                        z.write(filepath, arcname=single_file)