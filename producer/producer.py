import os, pika, logging, sys, argparse
from argparse import RawTextHelpFormatter
from time import sleep
import boto3
import botocore
from botocore.client import ClientError

def file_exists(bucket_name, file):
    try:
        s3.Object(bucket_name, file).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            exists = False
        else:
            raise e
    else:
        exists = True

    return exists


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger(__name__)

    examples = sys.argv[0] + " -p 5672 -s rabbitmq -i 'M56_tiles' -o 'M56_tile_detections'"
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                 description='Run producer.py',
                                 epilog=examples)
    parser.add_argument('-p', '--port', action='store', dest='port', help='The port to listen on.', required=False, default=5672)
    parser.add_argument('-s', '--server', action='store', dest='server', help='The RabbitMQ server.', required=False, default='rabbitmq')
    parser.add_argument('-i', '--in_folder', action='store', dest='in_folder', help='The bucket_name folder with image tiles to process', required=True)
    parser.add_argument('-r', '--repeat', action='store', dest='repeat', help='Number of times to repeat the message', required=False, default='30')

    args = parser.parse_args()

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

    try:
        s3.meta.client.head_bucket(Bucket=bucket_name)
    except ClientError:
        LOG.warning("Bucket %s doesn't exist or I don't have access to it " % bucket_name)
        LOG.warning('Here are all the buckets')
        for bucket in s3.buckets.all():
            LOG.warning(bucket.name)
        sys.exit(-1)

    bucket = s3.Bucket(bucket_name)

    # sleep a few seconds to allow RabbitMQ server to come up
    sleep(5)

    credentials = pika.PlainCredentials('guest', 'guest')
    parameters = pika.ConnectionParameters(args.server,
                                           int(args.port),
                                           '/',
                                           credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    q = channel.queue_declare('pc')
    q_name = q.method.queue

    # Turn on delivery confirmations
    channel.confirm_delivery()

    # list all keys in the input folder and only process those that
    # have no associated events.xml file
    for obj in bucket.objects.filter(Prefix=args.in_folder + '/'):
        LOG.info('{0}:{1}'.format(args.in_folder, obj.key))
        key = obj.key + '.events.xml'
        if '.jpg' in obj.key and not file_exists(bucket_name, key):
            if channel.basic_publish('', q_name, obj.key):
                LOG.info('Message has been delivered')
            else:
                LOG.warning('Message NOT delivered')

        sleep(2)

    connection.close()