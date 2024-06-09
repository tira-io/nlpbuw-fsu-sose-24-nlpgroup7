#!/bin/bash

# Define the Google Drive file ID and the desired output filename
FILE_ID="12px2bxEkuhaRXIXJUzubdd0gEjCkHiHh"
ZIP_FILE="trained_model.zip"


# Download the zip file using gdown
gdown --id $FILE_ID -O /code/$ZIP_FILE

# Unzip the downloaded file
unzip /code/$ZIP_FILE -d /code/

# Clean up the zip file
rm /code/$ZIP_FILE
