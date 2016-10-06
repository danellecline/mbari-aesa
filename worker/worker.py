import os, pika, logging, sys, argparse, time
from argparse import RawTextHelpFormatter
from botocore.client import ClientError
from time import sleep
import boto3
import botocore

# set up global variables for logging output to STDOUT
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# check environment variables required for connecting to S3 bucket
bucket_name = os.environ['S3_BUCKET_NAME']

if os.environ['AWS_ACCESS_KEY_ID'] is None:
    LOG.warninig('Missing required environment variable: AWS_ACCESS_KEY_ID')
    sys.exit(-1)
if os.environ['AWS_SECRET_ACCESS_KEY'] is None:
    LOG.warninig('Missing required environment variable: AWS_SECRET_ACCESS_KEY')
    sys.exit(-1)
if os.environ['AWS_DEFAULT_REGION'] is None:
    LOG.warninig('Missing required environment variable: AWS_DEFAULT_REGION')
    sys.exit(-1)
if bucket_name is None:
    LOG.warninig('Missing required environment variable: S3_BUCKET_NAME')
    sys.exit(-1)

# connect to Amazon S3
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

# define default output folder
out_folder = 'results'

try:
    s3.meta.client.head_bucket(Bucket=bucket_name)
except ClientError:
    LOG.warning("Bucket %s doesn't exist or I don't have access to it " % bucket_name)
    LOG.warning('Here are all the buckets')
    for bucket in s3.buckets.all():
        LOG.warning(bucket.name)
    sys.exit(-1)

# connect to bucket
bucket = s3.Bucket(bucket_name)

def on_message(channel, method_frame, header_frame, body):
    print method_frame.delivery_tag
    print body
    print
    LOG.info('Message has been received %s', body)
    channel.basic_ack(delivery_tag=method_frame.delivery_tag)

    if '.jpg' in body:
        s = body.split('/')
        root_dir = s[0]
        remote_file = s[1]
        local_file = remote_file
        root, ext = os.path.splitext(remote_file)
        LOG.info('Simulating long processing ......')

        sleep(60*5)

        LOG.info('Copying file %s from S3', body)
        s3_client.download_file(bucket_name, body, local_file)

        results_key = '{0}/{1}_results{2}'.format(out_folder, root, ext)
        LOG.info('Uploading %s', results_key)

        # Upload a new file
        data = open(local_file, 'rb')
        bucket.put_object(Key=results_key, Body=data)
        LOG.info('Uploading done')


if __name__ == '__main__':
    examples = sys.argv[0] + " -p 5672 -s rabbitmq "
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                 description='Run consumer.py',
                                 epilog=examples)
    parser.add_argument('-p', '--port', action='store', dest='port', help='The port to listen on.', required=False, default=5672)
    parser.add_argument('-s', '--server', action='store', dest='server', help='The RabbitMQ server.', required=False, default='rabbitmq')
    parser.add_argument('-o', '--out_folder', action='store', dest='out_folder', help='The bucket folder to store processed results', required=True)

    args = parser.parse_args()
    out_folder = args.out_folder

    # sleep a few seconds to allow RabbitMQ server to come up
    sleep(5)
    credentials = pika.PlainCredentials('guest', 'guest')
    parameters = pika.ConnectionParameters(args.server,
                                           int(args.port),
                                           '/',
                                           credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare('pc')
    channel.basic_consume(on_message, 'pc')

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    connection.close()
