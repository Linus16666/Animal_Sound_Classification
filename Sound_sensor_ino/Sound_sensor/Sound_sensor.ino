#include <LiquidCrystal.h>
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

void setup() {
  lcd.begin(16, 2);
  lcd.clear();
  lcd.print("Hallo");
  lcd.setCursor(0,1);
  lcd.print("Nicolas");
  Serial.begin(115200);

}

void loop() {
  int sound = analogRead(A1);
  Serial.println(sound);
}
