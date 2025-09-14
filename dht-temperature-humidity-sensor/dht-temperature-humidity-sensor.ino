#include <dht.h>

int DHT_data_pin = 10;
dht DHT;
void setup() {
  Serial.begin(9600);
}
void loop() {
  int readDHT = DHT.read11(DHT_data_pin);
  Serial.print("Temp: ");
  Serial.print(DHT.temperature);
  Serial.print("C");
  Serial.print("Humidity: ");
  Serial.print(DHT.humidity);
  Serial.println("%");
  delay(3000);
}