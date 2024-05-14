FROM fschlatt/natural-language-processing-exercises:0.0.1

#ADD language_identification_stopwords.py /code/language_identification_stopwords.py
ADD lang_identification_frequency_classify.py /code/lang_identification_frequency_classify.py
#ADD stopwords /code/stopwords
ADD frequencies /code/frequencies
COPY frequencies/* /code/frequencies

#ENTRYPOINT [ "python3", "/code/language_identification_stopwords.py" ]
ENTRYPOINT [ "python3", "/code/lang_identification_frequency_classify.py" ]
