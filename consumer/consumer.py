import os, pika, logging, sys, argparse, time
from argparse import RawTextHelpFormatter
from time import sleep
import boto3
import botocore


def on_message(channel, method_frame, header_frame, body):
    print method_frame.delivery_tag
    print body
    print
    LOG.info('Message has been received %s', body)
    channel.basic_ack(delivery_tag=method_frame.delivery_tag)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger(__name__)
    examples = sys.argv[0] + " -p 5672 -s rabbitmq "
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                 description='Run consumer.py',
                                 epilog=examples)
    parser.add_argument('-p', '--port', action='store', dest='port', help='The port to listen on.', required=False, default=5672)
    parser.add_argument('-s', '--server', action='store', dest='server', help='The RabbitMQ server.', required=False, default='rabbitmq')
    parser.add_argument('-i', '--in_folder', action='store', dest='in_folder', help='The bucket_name folder with image tiles to process', required=True)

    args = parser.parse_args()

    access_key = os.environ['AWS_ACCESS_KEY_ID']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
    region = os.environ['AWS_DEFAULT_REGION']
    bucket_name = os.environ['S3_BUCKET_NAME']

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


    # use Amazon S3
    s3 = boto3.resource('s3')
    client = boto3.client('s3')

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
