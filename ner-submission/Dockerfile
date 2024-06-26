# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1
COPY /model code/model

# Add requirements.txt
ADD requirements.txt /code/requirements.txt
# Install Python dependencies
RUN pip install -r /code/requirements.txt

ADD run.py /code/run.py

ENTRYPOINT [ "python3", "/code/run.py" ]