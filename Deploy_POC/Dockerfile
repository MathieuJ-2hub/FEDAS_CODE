# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /Load_and_give_results

# Add the current directory contents into the container at /app
ADD . /Load_and_give_results

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD streamlit run Stream.py