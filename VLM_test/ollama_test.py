import base64
from ollama import chat
from pathlib import Path

path = input('Please enter the path to the image: ').strip('\"')

image_path = Path(path)
if not image_path.exists():
    print("File does not exist")
else:
    img_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    response = chat(
        model='llava-phi3',
        messages=[
            {
                'role': 'user',
                'content': 'What is in this image? Be concise and answer in one word and predict its distance . Give output in s structured JSON format Like below: {"objects": ["dog", "cat"],"distance": "1 meter"}',
                'images': [img_data],
            }
        ],
    )

    print(response.message.content)

    response = chat(
        model='llava-phi3',
        messages=[
            {
                'role': 'user',
                'content': 'What is in this image? Be concise and answer in one word and predict its distance . Give output in s structured JSON format Like below: {"objects": ["dog", "cat"],"distance": "1 meter"}. You dont have to include any other data just the objects and the distance from the camera. Strictly follow the format.',
                'images': [img_data],
            }
        ],
    )

    print(response.message.content)