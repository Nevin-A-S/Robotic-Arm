from fastapi import FastAPI, WebSocket
import serial
import asyncio
import json
import csv
from datetime import datetime
import uvicorn

app = FastAPI()

# Configure serial connection to Arduino
ser = serial.Serial('COM3', 9600)  # Adjust port as needed

# Store for WebSocket connections
active_connections = []

class ServoController:
    def __init__(self):
        self.positions = [90] * 6  # Default positions
        
    async def set_servo(self, index: int, angle: int):
        command = f"SET,{index},{angle}\n"
        ser.write(command.encode())
        response = ser.readline().decode().strip()
        if response.startswith("OK"):
            self.positions[index] = angle
            return True
        return False
    
    async def get_positions(self):
        ser.write(b"GET\n")
        response = ser.readline().decode().strip()
        self.positions = [int(x) for x in response.split(',')]
        return self.positions
    
    def save_positions(self, name: str = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"servo_positions_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Servo', 'Angle', 'Name'])
            for i, pos in enumerate(self.positions):
                writer.writerow([i, pos, name if name else ''])
        return filename

controller = ServoController()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data['type'] == 'SET_SERVO':
                success = await controller.set_servo(data['index'], data['angle'])
                positions = await controller.get_positions()
                
                # Broadcast new positions to all connected clients
                for connection in active_connections:
                    await connection.send_json({
                        'type': 'POSITIONS_UPDATE',
                        'positions': positions
                    })
                    
            elif data['type'] == 'GET_POSITIONS':
                positions = await controller.get_positions()
                await websocket.send_json({
                    'type': 'POSITIONS_UPDATE',
                    'positions': positions
                })
                
            elif data['type'] == 'SAVE_POSITIONS':
                filename = controller.save_positions(data.get('name'))
                await websocket.send_json({
                    'type': 'SAVE_COMPLETE',
                    'filename': filename
                })
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        active_connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)