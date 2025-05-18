import requests
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--name", type=str, help="Run in register mode with username")

parser.add_argument("-i", action="store_true", help="Run in identification mode")

args = parser.parse_args()

if args.name:
    url = "http://localhost:8000/register"
    files = {
        'userId': (None, args.name),
        'file': ('image.jpg', open('/home/budhayan/Pictures/Camera/Photo from 2025-05-17 20-34-38.562295.jpeg', 'rb'))
    }
    response = requests.post(url, files=files)
    print(f"Result: {response.text}")

if args.i:
    url = "http://localhost:8000/identify"
        # 'file': ('image.jpg', open('/home/budhayan/Pictures/Camera/Photo from 2025-05-17 17-41-35.635702.jpeg', 'rb'))
    files = {
        'file': ('image.jpg', open('/home/budhayan/Pictures/Camera/Photo from 2025-05-17 20-34-38.562295.jpeg', 'rb'))

    }
    response = requests.post(url, files=files)
    # print(f"Time taken for identification is: {response.json()['time_taken']}")
    print(f"Result: {response.text}")