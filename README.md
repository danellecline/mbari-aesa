# mbari-aesa project

Code for processing AESA images in the cloud

## Prerequisites

- Install [Docker](https://docs.docker.com/installation/)
- Install [Compose](https://docs.docker.com/compose/install/)
- Create Amazon account [Amazon](http://www.amazon.com/)
- Create Docker account [Docker](http://www.docker.com/)

## Installation

Link Docker account to Amazon Web Services [Docker cloud] (https://cloud.docker.com/onboarding/)
[![ Docker cloud link ](https://github.com/danellecline/mbari-aesa/raw/master/img/docker-cloud-screenshot.png)]

Create Amazon bucket
[![ Amazon bucket link ](https://github.com/danellecline/mbari-aesa/raw/master/img/aws-bucket.png)]

Store bucket id, secret key, region and bucket name in .env file. See example in .env.example

    $ cp .env.example .env
    
    and add your credentials

Create docker stack

    $ 


## Local testing

Start container

    $ docker-compose up  
    
Start the management interface to see the message traffic
    
If running locally http://127.0.0.1:15672/
    