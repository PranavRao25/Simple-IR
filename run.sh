#!/bin/bash

# Running Tests on the source files
COMMAND="pytest"
DIRECTORY="tests/"

for file in "$DIRECTORY"/*; do
    if [ -f "$file" ]; then
        echo "Testing $file"
        "$COMMAND $file"

# Running the main.py
# python3 "src/main.py"