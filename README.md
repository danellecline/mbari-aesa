# mbari-aesa project

Code for processing AESA images in the cloud

## Prerequisites

- Install [Docker](https://docs.docker.com/installation/)
- Install [Compose](https://docs.docker.com/compose/install/)
- Create Amazon account [Amazon](http://www.amazon.com/)
- Connect Docker account [Amazon](http://www.amazon.com/)
- Link Docker account to Amazon Web Services [Docker cloud] (https://cloud.docker.com/onboarding/)

[![ Docker Cloud link ](https://github.com/danellecline/mbari-aesa/raw/master/img/docker-cloud-screenshot.png)]

## Installation


Set the environment variable RABBIT_HOST_IP. This should be the host IP you get using e.g. ifconfig.

    $ export RABBIT_HOST_IP=<your host IP - not localhost or 127.0.0.1>  

Start container

    $ docker-compose up  
    
If your environment variable is set incorrectly, you'll get something like

    $ pika.exceptions.AMQPConnectionError: Connection to 127.0.0.1:5672 failed: [Errno 111] Connection refused
    

## Start the management interface to see the message traffic
    
    http://${RABBIT_HOST_IP}:15672 or if running locally http://127.0.0.1:15672/
    