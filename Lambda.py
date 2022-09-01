
################################################################################
# FUNCTION 1 developing-ml-workflow-project-encode-image
################################################################################
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]

    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, "/tmp/image.png")

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

################################################################################
# FUNCTION 2 developing-ml-workflow-project-classify-image
################################################################################
import json
import boto3
import base64

ENDPOINT = "image-classification-2022-08-30-16-05-22-725"
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event["image_data"])

    response = runtime.invoke_endpoint(EndpointName=ENDPOINT,ContentType='image/png',Body=image)
    result = json.loads(response['Body'].read().decode('utf-8'))
    # We return the data back to the Step Function    
    event["inferences"] = result
    return {
        'statusCode': 200,
        'body': event
    }

################################################################################
# FUNCTION 3 developing-ml-workflow-project-check-inference-threshold
################################################################################
import json

THRESHOLD = .8

def lambda_handler(event, context):

    # Grab the inferences from the event
    inferences = event["inferences"]

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = True in [val > THRESHOLD for val in inferences]

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': event
    }