FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py

RUN conda install -n base -c conda-forge pot \
    && conda install -n base anaconda::gensim \
    && python -c "import gensim.downloader as api; api.load('word2vec-google-news-300')"

ENTRYPOINT [ "python3", "/code/run.py" ]