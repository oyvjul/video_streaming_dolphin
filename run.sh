#!/bin/bash
set -e

#
# USAGE: ./run.sh [--client hostname] [--server hostname]
#

CLIENT=localhost
SERVER=localhost
BUILDDIR=$(pwd)

FRONTEND=griff.mlab.no

CLIENT_CMD="./client"
SERVER_CMD="./server"

#Argument parsing
while [ $# -gt 0 ] ; do
    arg=$1
    shift

    case $arg in
        --client)
            CLIENT=$1
            shift
            ;;
        --server)
            SERVER=$1
            shift
            ;;
    esac
done

#Find node-ids for client and server
SERVER_NODE=$(ssh $FRONTEND /opt/DIS/sbin/disinfo get-nodeid -hostname $SERVER)
CLIENT_NODE=$(ssh $FRONTEND /opt/DIS/sbin/disinfo get-nodeid -hostname $CLIENT)

#Kill any old client and servers
ssh $CLIENT "killall -u $(whoami) client" &> /dev/null | true
ssh $SERVER "killall -u $(whoami) server" &> /dev/null | true

DATE=$(date -u +%Y%m%d-%H%M%S)
mkdir -p logs
#Run client and server
ssh $CLIENT "cd $BUILDDIR && $CLIENT_CMD -r $SERVER_NODE" |& tee logs/$DATE-client.log &
ssh $SERVER "cd $BUILDDIR && $SERVER_CMD -r $CLIENT_NODE" |& tee logs/$DATE-server.log &

wait

