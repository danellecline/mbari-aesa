#!/usr/bin/env bash
mkdir ~/.aws
# Insert credentials
cat >~/.aws/credentials <<_ACEOF
[default]
aws_access_key_id=${AWS_ACCESS_KEY_ID}
aws_secret_access_key=${AWS_SECRET_ACCESS_KEY}
region=${AWS_DEFAULT_REGION}
_ACEOF
exec "$@"