import asyncio
import websockets
import json

async def set_all_servos(angles):
    uri = "ws://localhost:8000/ws"  
    async with websockets.connect(uri) as websocket:

        for index, angle in enumerate(angles):
            command = {
                "type": "SET_SERVO",
                "index": index,
                "angle": angle
            }
            await websocket.send(json.dumps(command))
            print(f"Sent command for servo {index}: {command}")

            response = await websocket.recv()
            print("Received:", response)

if __name__ == "__main__":
    
    angles = [90, 45, 135, 180, 0, 90]
    asyncio.run(set_all_servos(angles))
