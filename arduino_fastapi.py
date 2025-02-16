from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import serial
import asyncio
import json
import csv
from datetime import datetime
import uvicorn
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure serial connection to Arduino
try:
    ser = serial.Serial(
        port='COM3',  # Change this to your Arduino port
        baudrate=9600,
        timeout=1
    )
    logger.info(f"Connected to Arduino on {ser.name}")
except Exception as e:
    logger.error(f"Failed to connect to Arduino: {e}")
    ser = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New client connected")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("Client disconnected")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

manager = ConnectionManager()

class ServoController:
    def __init__(self):
        self.positions = [90] * 6
        
    async def set_servo(self, index: int, angle: int) -> bool:
        if not ser:
            logger.error("No serial connection to Arduino")
            return False
            
        try:
            # Clear any pending data
            ser.reset_input_buffer()
            
            # Send command
            command = f"SET,{index},{angle}\n"
            logger.info(f"Sending command: {command.strip()}")
            ser.write(command.encode())
            ser.flush()
            
            # Wait for OK response
            response = ser.readline().decode().strip()
            logger.info(f"Initial response: {response}")
            
            if response.startswith("OK"):
                # Wait for position update
                position_response = ser.readline().decode().strip()
                logger.info(f"Position response: {position_response}")
                
                try:
                    # Update positions from Arduino's response
                    new_positions = [int(x) for x in position_response.split(',')]
                    if len(new_positions) == 6:
                        self.positions = new_positions
                except ValueError:
                    logger.error(f"Invalid position data: {position_response}")
                
                return True
            else:
                logger.error(f"Unexpected response: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting servo: {e}")
            return False
    
    async def get_positions(self) -> List[int]:
        if not ser:
            logger.error("No serial connection to Arduino")
            return self.positions
            
        try:
            # Clear any pending data
            ser.reset_input_buffer()
            
            # Send GET command
            ser.write(b"GET\n")
            ser.flush()
            
            # Wait for response
            response = ser.readline().decode().strip()
            logger.info(f"Got positions: {response}")
            
            try:
                # Parse positions
                new_positions = [int(x) for x in response.split(',')]
                if len(new_positions) == 6:
                    self.positions = new_positions
            except ValueError:
                logger.error(f"Invalid position data: {response}")
            
            return self.positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return self.positions
        
controller = ServoController()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            try:
                data = await websocket.receive_json()
                logger.info(f"Received message: {data}")
                
                if data['type'] == 'SET_SERVO':
                    success = await controller.set_servo(data['index'], data['angle'])
                    positions = await controller.get_positions()
                    
                    await manager.broadcast({
                        'type': 'POSITIONS_UPDATE',
                        'positions': positions,
                        'success': success
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
                    
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                continue
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    finally:
        if ser:
            ser.close()
            logger.info("Closed serial connection")