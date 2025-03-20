FROM python:3.11.11-slim-bullseye

ARG POETRY_VERSION=2.1.1


# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set the working directory to /app
WORKDIR /app

# Copy the poetry files into the container
COPY pyproject.toml poetry.lock /app/

# Install project dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-root

# Copy the rest of the application code into the container
COPY ./helikite /app

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g $GROUP_ID helikite && \
    useradd -u $USER_ID -g $GROUP_ID -m helikite
USER helikite

# Set the default command to run when the container starts
ENTRYPOINT ["python", "-u", "helikite.py"]
