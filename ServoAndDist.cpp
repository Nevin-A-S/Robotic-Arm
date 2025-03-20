#include <Servo.h>

// ----- Ultrasonic Sensor Variables -----
const int trigPin = 7;      // Trigger pin for the ultrasonic sensor
const int echoPin = 8;      // Echo pin for the ultrasonic sensor
long duration;
float distanceCm;
unsigned long lastSensorTime = 0;
const unsigned long SENSOR_INTERVAL = 500;  // Sensor measurement interval (ms)

// ----- Servo Control Variables -----
Servo servos[6];                    // Array to hold 6 servo objects
const int servoPins[6] = {3, 5, 6, 9, 10, 11};  // PWM pins for servos

float currentPos[6];                // Current positions for each servo (degrees)
int targetPos[6];                   // Target positions for each servo
unsigned long lastMoveTime = 0;
const int MOVE_INTERVAL = 15;       // Interval (in ms) between servo updates
const float SPEED = 0.5;            // Maximum degrees to move per update

// ----- Setup Function -----
void setup() {
  Serial.begin(9600);
  
  // Set up ultrasonic sensor pins
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  // Initialize servos: attach to pins and set starting position at 90 degrees
  for (int i = 0; i < 6; i++) {
    servos[i].attach(servoPins[i]);
    currentPos[i] = 90;
    targetPos[i] = 90;
    servos[i].write(round(currentPos[i]));
  }
  
  // Output the initial servo positions
  sendCurrentPositions();
}

// ----- Function to Send Current Servo Positions -----
void sendCurrentPositions() {
  String positions = "";
  for (int i = 0; i < 6; i++) {
    positions += String(round(currentPos[i]));
    if (i < 5) positions += ",";
  }
  Serial.println(positions);
}

// ----- Update Servo Positions Smoothly -----
void updateServoPositions() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastMoveTime >= MOVE_INTERVAL) {
    bool anyServoMoving = false;
    
    for (int i = 0; i < 6; i++) {
      // If the servo is not at its target position, move it gradually
      if (abs(currentPos[i] - targetPos[i]) > 0.1) {
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

// ----- Process Incoming Commands for Servos -----
// Expected commands:
//   SET,<servoIndex>,<angle>   - Set a specific servo's target angle
//   GET                        - Request current servo positions
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

// ----- Measure Distance Using the Ultrasonic Sensor -----
void measureDistance() {
  // Ensure trigger pin is LOW to get a clean measurement
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // Send a 10µs HIGH pulse to trigger the sensor
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Read the time (in microseconds) for which the echo pin is HIGH
  duration = pulseIn(echoPin, HIGH);

  // Calculate the distance in centimeters:
  // The speed of sound is ~0.034 cm/µs; divide by 2 for the round-trip.
  distanceCm = (duration * 0.034) / 2;
  
  // Output the distance to Serial Monitor
  Serial.print("Distance: ");
  Serial.print(distanceCm);
  Serial.println(" cm");
}

// ----- Main Loop -----
void loop() {
  // Process any incoming Serial commands for servo control
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    processCommand(command);
  }
  
  // Continuously update servo positions
  updateServoPositions();
  
  // Periodically measure and output the distance from the ultrasonic sensor
  if (millis() - lastSensorTime >= SENSOR_INTERVAL) {
    measureDistance();
    lastSensorTime = millis();
  }
}
