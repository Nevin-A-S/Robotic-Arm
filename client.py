import asyncio
import websockets
import json

async def send_message():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        message = {
            "type": "SET_SERVO",
            "index": 1,
            "angle": 150
        }
        await websocket.send(json.dumps(message))
        response = await websocket.recv()
        print("Response:", response)

asyncio.run(send_message())
