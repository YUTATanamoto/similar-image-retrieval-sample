FROM python:3.8
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt
CMD jupyter notebook --port 8888 --ip=0.0.0.0 --allow-root
