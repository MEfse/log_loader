FROM python:3.12

COPY ./requirements.txt ./requirements.txt 
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /project

#EXPOSE port

ENTRYPOINT [ "python", 'app.py']

#docker run --rm --name loader_data -d -p 5050:5050 