int led_pin = 13;
int led_pin2 = 12;

void setup() {
  pinMode(led_pin, OUTPUT);
  pinMode(led_pin2, OUTPUT);
}

void loop() {
  digitalWrite(led_pin, HIGH);
  digitalWrite(led_pin2, LOW);
  delay(1000);

  digitalWrite(led_pin, LOW);
  digitalWrite(led_pin2, HIGH);
  delay(1000);
}
