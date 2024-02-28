FROM tensorflow/tensorflow:latest-gpu 

# Install additional packages
RUN apt -y update
RUN apt -y upgrade
RUN apt install -y python3 python3-dev python3-pip

# Install any python packages you need
RUN pip install --upgrade pip
RUN pip install --ignore-installed shapely opencv-python-headless Flask
RUN pip install --ignore-installed tf-models-official


WORKDIR /app

COPY . /app
# alias
RUN echo 'alias [python](https://saturncloud.io/glossary/python)="python3" ' >> ~/.bashrc
RUN echo 'alias pip="pip3" ' >> ~/.bashrc

#Run the web server
EXPOSE 8000
ENV PYTHONPATH /app
CMD python3 /app/serve.py
