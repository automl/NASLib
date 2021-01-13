import argparse
import numpy as np
import torch
import boto3


def main(args):

    out_dir = args.out_dir
    zip_file = out_dir + '.zip'
    bucket = args.bucket
    folder = 'results_predictors/' + args.s3_folder + '/'

    s3_client = boto3.client('s3')
    print('uploading', zip_file, 'to', bucket, folder+zip_file)
    response = s3_client.upload_file(zip_file, bucket, folder + zip_file)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", type=str, default='run', help="Output directory")
    parser.add_argument("--bucket", type=str, default='realityengines.research', help="S3 Bucket")
    parser.add_argument("--s3_folder", type=str, default='pred', help="S3 folder")

    args = parser.parse_args()

    main(args)

