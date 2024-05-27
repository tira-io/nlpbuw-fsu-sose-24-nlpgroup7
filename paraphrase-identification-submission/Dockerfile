FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
ADD model.joblib /code/model.joblib
ADD cosineSimilarity.py /code/cosineSimilarity.py

RUN pip install spacy \
    && python3 -m spacy download en_core_web_md

ENTRYPOINT [ "python3", "/code/run.py" ]