#0 - Base Motor
#1 - Base Rotaion
#2 - Elbow Movement
#3 - Wrist Rotate
#4 - Wrist Movement
#5 - Gripper

import asyncio
import serial_asyncio

SERIAL_PORT = 'COM3'  # Change this to your Arduino's port (e.g., 'COM3' on Windows)
BAUDRATE = 9600

async def send_servo_command(writer, servo_index, angle):
    """
    Sends a command to set a specific servo to a given angle.
    The command format is: "servo_index:angle\n"
    """
    command = f"{servo_index}:{angle}\n"
    writer.write(command.encode())
    await writer.drain()
    print(f"Sent: {command.strip()}")

async def send_multiple_commands(writer, commands):
    """
    Sends multiple servo commands in one batch.
    'commands' should be a list of tuples: [(servo_index, angle), ...]
    Batch command format: "0:90,1:45,2:135,...\n"
    """
    command_str = ','.join([f"{idx}:{angle}" for idx, angle in commands]) + "\n"
    writer.write(command_str.encode())
    await writer.drain()
    print(f"Sent batch command: {command_str.strip()}")

async def main():
    # Open serial connection
    reader, writer = await serial_asyncio.open_serial_connection(url=SERIAL_PORT, baudrate=BAUDRATE)
    
    # Example 1: Send individual commands asynchronously
    individual_commands = [(0, 90), (1, 45), (2, 135), (3, 180), (4, 0), (5, 90)]
    tasks = [asyncio.create_task(send_servo_command(writer, idx, angle)) for idx, angle in individual_commands]
    await asyncio.gather(*tasks)
    
    # Wait a bit before sending the next set of commands
    await asyncio.sleep(2)
    
    # Example 2: Send a batch command to update all servos simultaneously
    batch_commands = [(0, 0), (1, 60), (2, 90), (3, 60), (4, 60), (5, 90)]
    await send_multiple_commands(writer, batch_commands)
    


    # Optionally, you can read responses from Arduino (if your Arduino sends any back)
    await asyncio.sleep(2)  # allow time for any responses
    while not reader.at_eof():
        line = await reader.readline()
        if line:
            print("Arduino:", line.decode().strip())
        else:
            break
    
    writer.close()
    await writer.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())
