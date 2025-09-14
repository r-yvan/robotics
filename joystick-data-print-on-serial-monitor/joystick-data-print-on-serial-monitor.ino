int sw_pin = 2;
int vrxpin = A0;
int vrypin = A1;

void setup() {
  // put your setup code here, to run once:

  serial.begin(9600);
  pinMode(sw_pin, INPUT_PULLUP);
}

void loop() {
  // put your main code here, to run repeatedly:
  
  int sw_val = digitalRead(sw_pin);
  int x_val = analogRead(vrxpin);
  int y_val = analogRead(vrypin);

  serial.print("sw value: ");
  serial.println(sw_val);
  serial.print("x value: ");
  serial.println(x_val);
  serial.print("y value: ");
  serial.println(y_val);
} 