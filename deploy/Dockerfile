FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Install additional packages
RUN apt -y update
RUN apt -y upgrade
RUN apt install -y python3 python3-dev python3-pip

# Install any python packages you need
RUN pip install --upgrade pip
RUN pip install shapely opencv-python-headless Flask
RUN pip install tf-models-official
RUN pip install --default-timeout=10000 tensorflow[and-cuda]

WORKDIR /app

COPY . /app
# alias
RUN echo 'alias [python](https://saturncloud.io/glossary/python)="python3" ' >> ~/.bashrc
RUN echo 'alias pip="pip3" ' >> ~/.bashrc

#Run the web server
EXPOSE 8000
ENV PYTHONPATH /app
CMD python3 /app/serve.py
