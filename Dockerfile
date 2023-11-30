FROM python:3.10-slim-bookworm
LABEL author="Henry Watkins"
LABEL author_contact="h.watkins@ucl.ac.uk"
LABEL description="neuroNLP dashboard docker image for the deployment of the neuroNLP neuroradiological reporting tool"
LABEL version="1.0"
WORKDIR /app
COPY . /app


COPY ./dist/*.whl /app/dist/

RUN pip install dist/neurocluster-0.0.1-py3-none-any.whl
RUN pip install dist/neurollm-0.0.1-py3-none-any.whl
RUN pip install dist/neuradicon-0.0.1-py3-none-any.whl
RUN pip install dist/neurodash-0.0.1-py3-none-any.whl

EXPOSE 8501
CMD ["streamlit", "run", "app.py" ,"--server.enableCORS=false","--server.maxUploadSize=1000" "--server.enableXsrfProtection=false"]
