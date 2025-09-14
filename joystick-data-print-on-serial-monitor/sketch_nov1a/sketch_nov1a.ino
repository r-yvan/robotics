int sw_pin = 2;
int vrxpin = A0;
int vrypin = A1;

void setup() {
  // put your setup code here, to run once:

  Serial.begin(9600);
  pinMode(sw_pin, INPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  
  int sw_val = digitalRead(sw_pin);
  int x_val = analogRead(vrxpin);
  int y_val = analogRead(vrypin);

  Serial.print("sw value: ");
  Serial.println(sw_val);
  Serial.print("x value: ");
  Serial.println(x_val);
  Serial.print("y value: ");
  Serial.println(y_val);
}