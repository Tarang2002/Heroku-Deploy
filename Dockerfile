# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container and install the required packages
COPY requirements.txt /app
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Set environment variables for the Back4App application
ENV PARSE_APPLICATION_ID <your_parse_application_id>
ENV PARSE_CLIENT_KEY <your_parse_client_key>
ENV PARSE_REST_API_KEY <your_parse_rest_api_key>
ENV PARSE_SERVER_URL https://parseapi.back4app.com/

# Expose the port on which the application will run
EXPOSE 8080

# Start the application
CMD ["python", "app.py"]
