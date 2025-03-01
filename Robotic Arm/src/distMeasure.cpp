// Define the pins for the ultrasonic sensor
const int trigPin = 9;
const int echoPin = 10;

void setup() {
  // Start the serial communication at 9600 baud
  Serial.begin(9600);
  
  // Set the trigPin as an OUTPUT and echoPin as an INPUT
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
}

void loop() {
  // Ensure the trigger pin is LOW for a brief moment
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  
  // Trigger the sensor by setting the trigger pin HIGH for 10 microseconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // Read the echo pin; pulseIn returns the time (in microseconds) that the pin stays HIGH
  long duration = pulseIn(echoPin, HIGH);
  
  // Calculate the distance (in cm)
  // Speed of sound is 343 m/s. The sound travels to the object and back, so we divide by 2.
  // Using the conversion factor: distance (cm) = (duration in µs) * 0.034 / 2.
  int distance = duration * 0.034 / 2;
  
  // Print the measured distance to the Serial Monitor
  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");
  
  // Wait half a second before the next reading
  delay(500);
}
