import os, pika, logging, sys, argparse
from argparse import RawTextHelpFormatter
from time import sleep
import boto3
from botocore.client import ClientError

if __name__ == '__main__':
    examples = sys.argv[0] + " -p 5672 -s rabbitmq -m 'Hello' "
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                 description='Run producer.py',
                                 epilog=examples)
    parser.add_argument('-p', '--port', action='store', dest='port', help='The port to listen on.')
    parser.add_argument('-s', '--server', action='store', dest='server', help='The RabbitMQ server.')
    parser.add_argument('-m', '--message', action='store', dest='message', help='The message to send', required=False, default='Hello')
    parser.add_argument('-r', '--repeat', action='store', dest='repeat', help='Number of times to repeat the message', required=False, default='30')

    args = parser.parse_args()
    if args.port == None:
        print "Missing required argument: -p/--port"
        sys.exit(1)
    if args.server == None:
        print "Missing required argument: -s/--server"
        sys.exit(1)

    access_key = os.environ['AWS_ACCESS_KEY_ID']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
    region = os.environ['AWS_DEFAULT_REGION']
    bucket_name = os.environ['S3_BUCKET_NAME']
    bucket_dir = os.environ['S3_BUCKET_DIR_NAME']

    if access_key is None:
        print "Missing required environment variable: AWS_ACCESS_KEY_ID"
        sys.exit(-1)
    if secret_key is None:
        print "Missing required environment variable: AWS_SECRET_ACCESS_KEY"
        sys.exit(-1)
    if region is None:
        print "Missing required environment variable: AWS_DEFAULT_REGION"
        sys.exit(-1)
    if bucket_name is None:
        print "Missing required environment variable: S3_BUCKET_NAME"
        sys.exit(-1)
    if bucket_dir is None:
        print "Missing required environment variable: S3_BUCKET_DIR_NAME"
        sys.exit(-1)

    # use Amazon S3
    s3 = boto3.resource('s3')
    client = boto3.client('s3')

    try:
        s3.meta.client.head_bucket(Bucket=bucket_name)
    except ClientError:
        print "Bucket %s doesn't exist or I don't have access to it " % bucket_name
        print "Here are all the buckets"
        for bucket in s3.buckets.all():
            print(bucket.name)
        sys.exit(-1)

    bucket = s3.Bucket(bucket_name)

    # create a directory under the main bucket to store the detection results
    response = client.put_object(
        Bucket=bucket_name,
        Body='',
        Key='detections/'
        )

    # sleep a few seconds to allow RabbitMQ server to come up
    sleep(5)

    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger(__name__)
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

    '''for i in range(0, int(args.repeat)):
        if channel.basic_publish('', q_name, args.message):
            LOG.info('Message has been delivered')
        else:
            LOG.warning('Message NOT delivered')

        sleep(2)'''

    # S3 list all keys with the prefix bucket_dir
    for obj in bucket.objects.filter(Prefix=bucket_dir + '/'):
        print('{0}:{1}'.format(bucket.name, obj.key))
        if '.jpg' in obj.key:
            if channel.basic_publish('', q_name, obj.key):
                LOG.info('Message has been delivered')
            else:
                LOG.warning('Message NOT delivered')

        sleep(2)

    connection.close()