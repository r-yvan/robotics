#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27, 16, 2);
int trigPin = 9;
int echoPin = 8; 
int ledpingreen = 6;
int ledpinred =  5;
float duration_us, distance_cm;
void setup() {
 lcd.init();
 lcd.backlight(); 
 pinMode(trigPin, OUTPUT); 
 pinMode(echoPin, INPUT); 
 pinMode(ledpingreen, OUTPUT);
 pinMode(ledpinred, OUTPUT);

}
void loop() {
 digitalWrite(trigPin, HIGH);
 delayMicroseconds(10);
 digitalWrite(trigPin, LOW);
 duration_us = pulseIn(echoPin, HIGH);
 distance_cm = 0.017 * duration_us;
 lcd.clear();
 lcd.setCursor(0, 0);
 lcd.print("D:");
 lcd.print(distance_cm);
 if (distance_cm < 10){
  lcd.print("high risk");
  digitalWrite(ledpinred,HIGH);
  digitalWrite(ledpingreen,LOW);
 }else{
  lcd.print("no risk");
  digitalWrite(ledpingreen,HIGH);
  digitalWrite(ledpinred,LOW);
 }
  delay(500);
}
