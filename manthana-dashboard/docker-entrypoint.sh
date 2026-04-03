#!/bin/sh
set -e
ORTHANC_AUTH_BASIC=$(printf '%s' "${ORTHANC_USER}:${ORTHANC_PASS}" | base64 | tr -d '\n')
export ORTHANC_AUTH_BASIC
# Only substitute ORTHANC_AUTH_BASIC — a full envsubst would wipe nginx $uri, $host, etc.
envsubst '${ORTHANC_AUTH_BASIC}' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf
exec nginx -g 'daemon off;'
