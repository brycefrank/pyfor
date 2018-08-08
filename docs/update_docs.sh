#!/bin/bash
# This script autoregenerates pyfor's documentation for updates to master.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
sphinx-apidoc -f -o source/ $DIR/../pyfor --separate
cd $DIR
make html
cd -
