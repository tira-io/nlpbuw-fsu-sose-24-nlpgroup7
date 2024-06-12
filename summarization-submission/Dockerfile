FROM fschlatt/natural-language-processing-exercises:0.0.1

# Install gdown to download files from Google Drive
RUN pip install gdown

# Install unzip
RUN apt-get update && apt-get install -y unzip

# Add the run.py script to the container
ADD run_cuda.py /code/run.py

# Add the script to download and unzip the model from Google Drive
ADD download_and_unzip.sh /code/download_and_unzip.sh
RUN chmod +x /code/download_and_unzip.sh

# Run the download and unzip script
RUN /code/download_and_unzip.sh

# Add requirements.txt
ADD requirements.txt /code/requirements.txt
# Install Python dependencies
RUN pip install -r /code/requirements.txt

ENTRYPOINT [ "python3", "/code/run.py" ]
