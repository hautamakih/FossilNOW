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

RUN chgrp root data_processed/mf/prob_True/emb_sites.npy && chmod 770 data_processed/mf/prob_True/emb_sites.npy

RUN chgrp root data_processed/mf/prob_True/emb_genera.npy && chmod 770 data_processed/mf/prob_True/emb_genera.npy

RUN chgrp root data_processed/mf/prob_True/model.pt && chmod 770 data_processed/mf/prob_True/model.pt

RUN chgrp root data_processed/mf/prob_False/emb_sites.npy && chmod 770 data_processed/mf/prob_False/emb_sites.npy

RUN chgrp root data_processed/mf/prob_False/emb_genera.npy && chmod 770 data_processed/mf/prob_False/emb_genera.npy

RUN chgrp root data_processed/mf/prob_False/model.pt && chmod 770 data_processed/mf/prob_False/model.pt

CMD ["gunicorn", "-b", "0.0.0.0:8050", "--reload", "app:server"]