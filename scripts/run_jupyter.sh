#!/bin/bash -
#===============================================================================
#
#          FILE: run_jupyter.sh
#
#         USAGE: ./run_jupyter.sh
#
#   DESCRIPTION: Script to start Jupyter lab and display the right url on server.
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: Axel Fahy (axel@fahy.net),
#  ORGANIZATION:
#       CREATED: 01. 05. 19 15:07
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error.

declare -r HOST=`hostname`

echo -e "Possible Jupyter binaries: $(whereis jupyter)"
echo -e "Using Jupyter from $(which jupyter)"

echo -e "Looking for an available port..."

declare port=8888

for i in $(seq $port 8950); do
    # Check if the port is already used.
    echo -e "Checking port: $i"
    ss -tlnp | grep -q $i &> /dev/null
    if [ $? -ne 0 ]; then
        port=$i
        echo -e "Using port: $port"
        break
    fi
done

jupyter lab --ip=0.0.0.0 --NotebookApp.port=$port --NotebookApp.custom_display_url=http://$HOST:$port

