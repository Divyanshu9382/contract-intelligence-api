# Start from an official Python 3.11 image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file first and install dependencies
# This leverages Docker caching - it only re-installs if requirements.txt changes
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Now copy the rest of your application code
COPY ./app /code/app

# Command to run your application using uvicorn
# It will run the 'app' object from the 'app.main' module
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
