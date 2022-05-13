import os
import boto3
from botocore.exceptions import NoCredentialsError

ACCESS_KEY = 'AKIA5EGXCASQVUR4SYNP'
SECRET_KEY = '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'
BUCKET = 'engine-reports'

def download_files(system_serial, destination):
    client = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)
    prefix = "{}/".format(system_serial)
    local = destination
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket': BUCKET,
        'Prefix': prefix,
    }

    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')

    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))

    print("\nFound {} item(s) for serial {}".format(len(keys), system_serial))

    for k in keys:
        dest_pathname = os.path.join(local, k)

        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))

        print("-> Downloading '{}/{}'".format(BUCKET, k))
        client.download_file(BUCKET, k, dest_pathname)

if __name__ == '__main__':
    download_files("180733444100898", "/Users/galsabag/Downloads")