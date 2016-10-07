# mbari-aesa project

Code for processing AESA images in the cloud

## Prerequisites

- Install [Docker](https://docs.docker.com/installation/)
- Install [Compose](https://docs.docker.com/compose/install/)
- Create Amazon account [Amazon](http://www.amazon.com/)
- Create Docker ID [Docker](http://hub.docker.com/)

## Installation

Link Docker account to Amazon Web Services [Docker cloud] (https://cloud.docker.com/onboarding/)
[![ Docker cloud link ](https://github.com/danellecline/mbari-aesa/raw/master/img/docker-cloud-screenshot.png)]

Create Amazon bucket
[![ Amazon bucket link ](https://github.com/danellecline/mbari-aesa/raw/master/img/aws-bucket.png)]

Store bucket id, secret key, region and bucket name in .env file. See example in .env.example

    $ cp .env.example .env
    
    and add your credentials

Cut and paste docker stack files docker-cloud.yml to Docker cloud account. TODO: how to automate this ?

[![ Cloud stack link ](https://github.com/danellecline/mbari-aesa/raw/master/img/docker-stack-screenshot.png)]

## Local testing

Start container

    $ docker-compose up  
    
Start the management interface to see the message traffic
    
If running locally http://127.0.0.1:15672/
    
To make change to app code, only need to rebuild service and deploy that service, e.g. to just redeploy worker

    $ docker-compose build worker
    $ docker-compose up --no-deps -d worker