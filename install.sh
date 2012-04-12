#!/bin/bash

#
# Install script. Sets the path to the IDAM library, then runs the Python install script
# 
# Usage:
#
# ./install.sh   [ /path/to/idam/ ]
# 
# The path is optional, and if not given will be asked for
# 

if [ "$1" == "" ]; then
	# Ask the user for the path
	read -p  "Enter path to IDAM library source code: " path
else
	path=$1
fi

oldpath="/home/ben/codes/idam"

sed "s|$oldpath|$path|" install.py > setup.py

python setup.py install

