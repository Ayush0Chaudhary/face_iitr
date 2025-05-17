import requests

url = "http://localhost:8000/identify"
files = {
    'file': ('image.jpg', open('/home/budhayan/Pictures/Camera/Photo from 2025-05-17 17-41-35.635702.jpeg', 'rb'))
}
response = requests.post(url, files=files)

print(f"Time taken for identification is: {response.json()['time_taken']}")

print(f"Result: {response.text}")