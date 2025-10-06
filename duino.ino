#include <Servo.h>

Servo servos[5];
int pins[5] = {3,5,6,9,10};
int angle[5] = {90,90,90,90,90};

void setup() {
  Serial.begin(115200);
  for (int i=0;i<5;i++) servos[i].attach(pins[i]);
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;


    float vals[5];
    int count = sscanf(line.c_str(), "%f,%f,%f,%f,%f",
                       &vals[0], &vals[1], &vals[2], &vals[3], &vals[4]);
    if (count == 5) {
      for (int i=0;i<5;i++) {
        int a = (int)(vals[i]*180.0);        // map 0–1 → 0–180°
        if (a < 0) a = 0;
        if (a > 180) a = 180;
        servos[i].write(a);
      }
    }
  }
}
