from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import serial
import asyncio
import json
import csv
from datetime import datetime
import uvicorn
import logging
from typing import List
import os
import re

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

class RobotController:
    def __init__(self):
        self.positions = [90] * 6
        self.distance = 0.0
        
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
                # Wait for movement to complete by reading position updates
                while True:
                    position_response = ser.readline().decode().strip()
                    logger.info(f"Position response: {position_response}")
                    
                    # Check if it's a distance reading
                    if position_response.startswith("DIST:"):
                        try:
                            dist_value = float(position_response[5:])
                            self.distance = dist_value
                            # Continue waiting for position update
                            continue
                        except ValueError:
                            logger.error(f"Invalid distance data: {position_response}")
                    
                    try:
                        # Update positions from Arduino's response
                        new_positions = [int(x) for x in position_response.split(',')]
                        if len(new_positions) == 6:
                            self.positions = new_positions
                            # Check if target position is reached
                            if abs(new_positions[index] - angle) <= 1:
                                break
                    except ValueError:
                        logger.error(f"Invalid position data: {position_response}")
                        # Don't break, try to read next line
                
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
    
    async def get_distance(self) -> float:
        """Request and get the current distance reading from the sensor"""
        if not ser:
            logger.error("No serial connection to Arduino")
            return self.distance
            
        try:
            # Clear any pending data
            ser.reset_input_buffer()
            
            # Send GETDIST command
            ser.write(b"GETDIST\n")
            ser.flush()
            
            # Wait for response
            response = ser.readline().decode().strip()
            logger.info(f"Got distance: {response}")
            
            # Expected format: "DIST:123.45"
            if response.startswith("DIST:"):
                try:
                    self.distance = float(response[5:])
                except ValueError:
                    logger.error(f"Invalid distance data: {response}")
            
            return self.distance
            
        except Exception as e:
            logger.error(f"Error getting distance: {e}")
            return self.distance

    async def reset_positions(self) -> bool:
        """Reset all servos to 90 degrees"""
        success = True
        for i in range(6):
            if not await self.set_servo(i, 90):
                success = False
        return success

    def save_positions(self, name: str) -> str:
        """Save current positions to CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"positions_{name}_{timestamp}.csv"
        
        try:
            os.makedirs('saved_positions', exist_ok=True)
            filepath = os.path.join('saved_positions', filename)
            
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Name', 'Timestamp', 'Distance_cm'] + [f'Servo_{i+1}' for i in range(6)])
                writer.writerow([name, timestamp, self.distance] + self.positions)
                
            logger.info(f"Saved positions to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
            return ""

async def monitor_distance():
    """Background task to monitor distance readings and broadcast updates"""
    while True:
        if ser:
            try:
                # Read lines from the serial port
                if ser.in_waiting > 0:
                    line = ser.readline().decode().strip()
                    
                    # Check if it's a distance reading
                    if line.startswith("DIST:"):
                        try:
                            dist_value = float(line[5:])
                            controller.distance = dist_value
                            
                            # Broadcast the distance update
                            await manager.broadcast({
                                'type': 'DISTANCE_UPDATE',
                                'distance': dist_value
                            })
                        except ValueError:
                            logger.error(f"Invalid distance data: {line}")
            except Exception as e:
                logger.error(f"Error reading sensor data: {e}")
        
        # Short delay to avoid hogging CPU
        await asyncio.sleep(0.1)
        
controller = RobotController()

@app.on_event("startup")
async def startup_event():
    # Start the background task to monitor sensor readings
    asyncio.create_task(monitor_distance())

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
                    positions = controller.positions  # Use stored positions
                    
                    await manager.broadcast({
                        'type': 'POSITIONS_UPDATE',
                        'positions': positions,
                        'distance': controller.distance,
                        'success': success
                    })
                    
                elif data['type'] == 'GET_POSITIONS':
                    positions = await controller.get_positions()
                    await websocket.send_json({
                        'type': 'POSITIONS_UPDATE',
                        'positions': positions,
                        'distance': controller.distance
                    })
                
                elif data['type'] == 'GET_DISTANCE':
                    distance = await controller.get_distance()
                    await websocket.send_json({
                        'type': 'DISTANCE_UPDATE',
                        'distance': distance
                    })

                elif data['type'] == 'RESET_POSITIONS':
                    success = await controller.reset_positions()
                    positions = controller.positions
                    await manager.broadcast({
                        'type': 'POSITIONS_UPDATE',
                        'positions': positions,
                        'distance': controller.distance,
                        'success': success
                    })
                    
                elif data['type'] == 'SAVE_POSITIONS':
                    filename = controller.save_positions(data.get('name', ''))
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