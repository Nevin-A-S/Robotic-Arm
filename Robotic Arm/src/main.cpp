#include <Arduino.h>
#include <Servo.h>

Servo myServo;

void setup() {
  myServo.attach(9);
}

void loop() {
  // Move from 0 to 180 degrees using 5-degree increments for a faster movement
  for (int angle = 0; angle <= 180; angle += 1) {
    myServo.write(angle);
    delay(5);  // You can still adjust delay for smoother transitions
  }
  
  // Move back from 180 to 0 degrees
  for (int angle = 180; angle >= 0; angle -= 1) {
    myServo.write(angle);
    delay(5);
  }
}
