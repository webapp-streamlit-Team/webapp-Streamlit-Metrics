FROM ubuntu:latest

WORKDIR C:\Users\chris\Desktop\webapp

ARG LANG='en_us.UTF-8'

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        apt-utils \
        locales \
        python3-pip \
        python3-yaml \
        rsyslog systemd systemd-cron sudo \
    && apt-get clean

RUN pip3 install --upgrade pip 

RUN pip3 install streamlit

RUN pip3 install plotly

RUN pip3 install umap-learn

RUN pip3 install numpy

RUN pip3 install pandas

RUN pip3 install datasets

RUN pip3 install scikit-learn

RUN pip3 install openpyxl

COPY / ./ 

CMD ["streamlit", "run", "streamlit.py"]