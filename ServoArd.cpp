#include <Servo.h>

Servo servos[6];  // Array to hold 6 servo objects
const int servoPins[6] = {3, 5, 6, 9, 10, 11};  // PWM pins for servos

// Current and target positions for each servo
float currentPos[6];
int targetPos[6];
unsigned long lastMoveTime = 0;

// Movement parameters
const int MOVE_INTERVAL = 15;    // Time between position updates (ms)
const float SPEED = 0.5;         // Degrees per movement interval

void setup() {
  Serial.begin(9600);
  
  // Initialize each servo
  for (int i = 0; i < 6; i++) {
    servos[i].attach(servoPins[i]);
    currentPos[i] = 90;  // Start at middle position
    targetPos[i] = 90;
    servos[i].write(round(currentPos[i]));
  }

  // Send initial positions
  sendCurrentPositions();
}

void sendCurrentPositions() {
  String positions = "";
  for (int i = 0; i < 6; i++) {
    positions += String(round(currentPos[i]));
    if (i < 5) positions += ",";
  }
  Serial.println(positions);
}

void updateServoPositions() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastMoveTime >= MOVE_INTERVAL) {
    bool anyServoMoving = false;
    
    for (int i = 0; i < 6; i++) {
      if (abs(currentPos[i] - targetPos[i]) > 0.1) {  // Check if servo needs to move
        float diff = targetPos[i] - currentPos[i];
        float moveAmount = constrain(diff, -SPEED, SPEED);
        
        currentPos[i] += moveAmount;
        servos[i].write(round(currentPos[i]));
        anyServoMoving = true;
      }
    }
    
    lastMoveTime = currentTime;
    
    if (anyServoMoving) {
      sendCurrentPositions();
    }
  }
}

void processCommand(String command) {
  if (command.startsWith("SET")) {
    int firstComma = command.indexOf(',');
    int secondComma = command.indexOf(',', firstComma + 1);
    
    if (firstComma != -1 && secondComma != -1) {
      int servoIndex = command.substring(firstComma + 1, secondComma).toInt();
      int angle = command.substring(secondComma + 1).toInt();
      
      if (servoIndex >= 0 && servoIndex < 6 && angle >= 0 && angle <= 180) {
        targetPos[servoIndex] = angle;
        Serial.print("OK,");
        Serial.println(servoIndex);
        return;
      }
    }
    Serial.println("ERROR,Invalid command format");
  }
  else if (command == "GET") {
    sendCurrentPositions();
  }
  else {
    Serial.println("ERROR,Unknown command");
  }
}

void loop() {
  // Process any incoming commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();  // Remove any whitespace
    processCommand(command);
  }
  
  // Update servo positions
  updateServoPositions();
}