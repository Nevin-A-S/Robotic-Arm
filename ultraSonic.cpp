// Define sensor pins
const int trigPin = 7;  // Trigger pin on Arduino Uno
const int echoPin = 8;  // Echo pin on Arduino Uno

long duration;
float distanceCm;

void setup() {
  Serial.begin(9600);      // Initialize Serial Monitor at 9600 baud rate
  pinMode(trigPin, OUTPUT); // Set trigger pin as output
  pinMode(echoPin, INPUT);  // Set echo pin as input
}

void loop() {
  // Ensure the trigger pin is LOW for a stable reading
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // Send a 10µs pulse to the trigger pin to start the measurement
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Read the time (in microseconds) for which the echo pin is HIGH
  duration = pulseIn(echoPin, HIGH);

  // Calculate the distance:
  // Speed of sound is ~0.034 cm/µs; dividing by 2 accounts for the round-trip distance.
  distanceCm = (duration * 0.034) / 2;

  // Output the distance to the Serial Monitor
  Serial.print("Distance: ");
  Serial.print(distanceCm);
  Serial.println(" cm");

  delay(500); // Short delay before taking the next measurement
}
