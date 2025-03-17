from client_interactio import set_all_servos
from liveServoPosition import get_servo_positions
import asyncio

gripperLock = 90
gripperFree = 30
pickFront = [90, 45, 135, 180, gripperFree, 90]
pickUpFront = [90, 45, 135, 180, gripperLock, 90]
placeLeft = [90, 45, 135, 180, gripperLock, 90]
placeRight = [90, 45, 135, 180, gripperLock, 90]

def getGripperFree():
    postions = asyncio.run(get_servo_positions())
    postions[4] = gripperFree
    return postions

async def move_right():
    await set_all_servos(pickFront)
    await set_all_servos(pickUpFront)
    await set_all_servos(placeRight)
    await set_all_servos(getGripperFree())

async def moveLeft():
    await set_all_servos(pickFront)
    await set_all_servos(pickUpFront)
    await set_all_servos(placeLeft)
    await set_all_servos(getGripperFree())

if __name__ == "__main__":
    asyncio.run(move_right())