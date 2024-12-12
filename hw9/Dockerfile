# Use the provided image as the base
FROM agrigorev/model-2024-hairstyle:v3

# Set the working directory
WORKDIR /app

# Install required Python libraries
RUN pip install --no-cache-dir flask pillow tensorflow

# Copy the Lambda function code into the image
COPY lambda_function.py /app/lambda_function.py

# Expose the default Flask port
EXPOSE 5000

# Command to run the Lambda function
CMD ["python", "lambda_function.py"]
