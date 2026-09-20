#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define SERIAL_BAUD     115200
#define NUM_CHANNELS    12
#define PAYLOAD_SIZE    (NUM_CHANNELS * 2)   // 12x uint16 (big-endian) = 24 bytes
#define MAX_PACKET_SIZE 40                   // COBS of 24 non-zero bytes fits easily
#define ERROR_BYTE      0xFF
#define LED_PIN         13
#define LED_DURATION_MS 1000

static unsigned long ledOffAt = 0;

// Last commanded pulse width per PCA9685 channel (0..11). Neutral on boot.
static uint16_t chan[NUM_CHANNELS] = {
  1500, 1500, 1500, 1500, 1500, 1500,
  1500, 1500, 1500, 1500, 1500, 1500
};

// ---------------------------------------------------------------------------
// Minimal COBS decoder
// ---------------------------------------------------------------------------
static uint8_t cobsDecode(const uint8_t *src, uint8_t srcLen,
                           uint8_t *dst, uint8_t dstMaxLen) {
  if (srcLen == 0) return 0;

  uint8_t dstIdx = 0;
  uint8_t srcIdx = 0;

  while (srcIdx < srcLen) {
    uint8_t code = src[srcIdx++];
    if (code == 0) return 0;

    uint8_t numLiterals = code - 1;
    if (srcIdx + numLiterals > srcLen) return 0;
    for (uint8_t i = 0; i < numLiterals; i++) {
      if (src[srcIdx] == 0) return 0;
      if (dstIdx >= dstMaxLen) return 0;
      dst[dstIdx++] = src[srcIdx++];
    }

    if (srcIdx < srcLen) {
      if (dstIdx >= dstMaxLen) return 0;
      dst[dstIdx++] = 0x00;
    }
  }

  return dstIdx;
}

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

static void writeAllChannels() {
  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    pwm.writeMicroseconds(i, chan[i]);
  }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(SERIAL_BAUD);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(50);

  writeAllChannels();
}

// ---------------------------------------------------------------------------
// Loop
// ---------------------------------------------------------------------------
void loop() {
  static uint8_t buf[MAX_PACKET_SIZE];
  static uint8_t bufLen = 0;

  // Continuously re-assert last known position
  writeAllChannels();

  // Turn LED off once duration has elapsed
  if (ledOffAt != 0 && millis() >= ledOffAt) {
    digitalWrite(LED_PIN, LOW);
    ledOffAt = 0;
  }

  while (Serial.available()) {
    uint8_t b = (uint8_t)Serial.read();

    if (b == 0x00) {
      if (bufLen == 0) continue;

      uint8_t decoded[PAYLOAD_SIZE];
      uint8_t decodedLen = cobsDecode(buf, bufLen, decoded, sizeof(decoded));

      if (decodedLen != PAYLOAD_SIZE) {
        Serial.write(ERROR_BYTE);
      } else {
        // 12x uint16, big-endian, channel order 0..11
        for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
          chan[i] = ((uint16_t)decoded[i * 2] << 8) | decoded[i * 2 + 1];
        }
        writeAllChannels();

        digitalWrite(LED_PIN, HIGH);
        ledOffAt = millis() + LED_DURATION_MS;
      }

      bufLen = 0;

    } else {
      if (bufLen >= MAX_PACKET_SIZE) {
        Serial.write(ERROR_BYTE);
        bufLen = 0;
      } else {
        buf[bufLen++] = b;
      }
    }
  }
}
