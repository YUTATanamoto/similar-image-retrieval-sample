FROM python:3.8
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt
COPY launch-jupyter.sh /launch-jupyter.sh
USER root
RUN chmod 755 /launch-jupyter.sh
CMD ["/launch-jupyter.sh"]
