import asyncio
import websockets
import json

async def get_servo_positions():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:

        command = {
            "type": "GET_POSITIONS"
        }
        await websocket.send(json.dumps(command))
        print("Requested servo positions...")

        response = await websocket.recv()
        data = json.loads(response)
        if data.get('type') == 'POSITIONS_UPDATE':
            positions = data.get('positions')
            # distance = data.get('distance')
            print("Current servo positions:", positions)
            # print("Current distance:", distance)
            return positions
        else:
            print("Unexpected response:", data)

if __name__ == "__main__":
    asyncio.run(get_servo_positions())
