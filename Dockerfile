FROM python:3.10-slim

WORKDIR /usr/src/app

RUN chmod -R 777 /usr/src/app

EXPOSE 8050

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src .

RUN chgrp root data_processed && chmod 770 data_processed

RUN chgrp root data_processed/encoding && chmod 770 data_processed/encoding

RUN chgrp root data_processed/encoding/ordinal_enc_genus.json && chmod 770 data_processed/encoding/ordinal_enc_genus.json

RUN chgrp root data_processed/encoding/ordinal_enc_site.json && chmod 770 data_processed/encoding/ordinal_enc_site.json

CMD ["gunicorn", "-b", "0.0.0.0:8050", "--reload", "app:server"]