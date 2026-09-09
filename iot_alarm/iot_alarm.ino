#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

const char* ssid = "Pandu";
const char* password = "pandu280212";

const int RELAY_PIN = 5; // D1 di NodeMCU

ESP8266WebServer server(80);

// --- PERBAIKAN LOGIKA RELAY ---
// Ubah logika jika relay kamu ternyata Active-HIGH
const int RELAY_NYALA = HIGH; // Sinyal HIGH untuk trigger relay ALARM (Nyala)
const int RELAY_MATI = LOW;   // Sinyal LOW untuk matikan relay (Standby/Off)

void matikanLampu() {
  digitalWrite(RELAY_PIN, RELAY_MATI);
}

void setup() {
  Serial.begin(115200);
  delay(10);

  pinMode(RELAY_PIN, OUTPUT);
  matikanLampu(); 

  Serial.println();
  Serial.print("Menyambungkan ke WiFi: ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi terhubung!");
  Serial.print("====> CATAT IP ADDRESS INI: ");
  Serial.println(WiFi.localIP());

  // ROUTE HTTP
  server.on("/merah", []() {
    Serial.println("TERIMA SINYAL: KARDUS NYASAR (NG) -> RELAY AKTIF!");
    digitalWrite(RELAY_PIN, RELAY_NYALA);
    server.send(200, "text/plain", "OK - Alarm Merah Nyala");
    
    delay(2000); // Alarm menyala selama 2 detik
    matikanLampu();
  });

  server.on("/hijau", []() {
    Serial.println("TERIMA SINYAL: KARDUS MATCH -> RELAY STANDBY (OFF)!");
    matikanLampu(); 
    server.send(200, "text/plain", "OK - Standby");
  });

  server.begin();
}

void loop() {
  server.handleClient();
}