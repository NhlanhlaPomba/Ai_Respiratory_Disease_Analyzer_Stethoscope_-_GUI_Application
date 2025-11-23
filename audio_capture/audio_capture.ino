/*
 * ESP32-WROOM-32 MAX4466 Microphone Audio Streaming - OPTIMIZED
 * 
 * Hardware Setup:
 * - ESP32-WROOM-32 Development Board
 * - MAX4466 Electret Microphone Amplifier with Adjustable Gain
 * - Connect VCC to ESP32 3.3V
 * - Connect GND to ESP32 GND
 * - Connect OUT to ESP32 GPIO34 (ADC1_CH6)
 * 
 * MAX4466 Advantages:
 * - High-quality electret microphone element
 * - Built-in amplifier with adjustable gain (25x to 125x)
 * - DC-blocked output (already centered at VCC/2)
 * - Low noise design
 * - Wide frequency response (20Hz-20kHz)
 * - Rail-to-rail output
 * 
 * This code is optimized for respiratory sound analysis
 * 
 * Author: Respiratory Disease Classification System
 * Date: 2025
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

// Pin Configuration
const int MIC_PIN = 34;            // GPIO34 (ADC1_CH6) - MAX4466 OUT pin
const int LED_PIN = 2;             // Built-in LED

// Audio Configuration
const long SAMPLE_RATE = 16000;    // 16 kHz - HIGHER for MAX4466 quality!
                                   // MAX4466 supports up to 20kHz
const int SAMPLES_PER_CHUNK = 128; // Optimized buffer size
const long BAUD_RATE = 230400;     // HIGHER baud rate for 16kHz sampling
                                   // Options: 115200, 230400, 460800, 921600

// ADC Configuration for ESP32 + MAX4466
const int ADC_RESOLUTION = 12;     // 12-bit ADC (0-4095)
const int ADC_CENTER = 2048;       // Center value for 12-bit
const int ADC_MAX = 4095;

// MAX4466 Specific Settings
// MAX4466 output is already DC-blocked and centered at VCC/2
// This means the signal oscillates around 1.65V (ADC reading ~2048)
const int MAX4466_GAIN = 50;       // Approximate gain setting (adjust pot)
                                   // 25x (min) to 125x (max)

// Advanced Audio Processing
const bool ENABLE_DC_FILTER = true;      // High-pass filter
const bool ENABLE_NOISE_GATE = true;     // Remove background noise
const bool ENABLE_AGC = false;           // Auto gain control (optional)

// Noise gate threshold (0-4095)
const int NOISE_THRESHOLD = 50;          // Adjust based on environment

// Timing
unsigned long samplePeriod;
unsigned long lastSampleTime = 0;

// State
bool isRecording = false;
bool ledState = false;

// Buffer
int16_t audioBuffer[SAMPLES_PER_CHUNK];
int bufferIndex = 0;

// Audio processing variables
float dcOffset = 0.0;                    // For DC offset removal
int32_t agcLevel = 32768;                // For automatic gain control
int16_t peakValue = 0;                   // For peak detection

// Performance monitoring
unsigned long samplesProcessed = 0;
unsigned long lastStatsTime = 0;

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  // Initialize Serial with HIGHER baud rate
  Serial.begin(BAUD_RATE);
  
  // Setup pins
  pinMode(MIC_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  
  // Configure ADC for MAX4466
  analogReadResolution(12);              // 12-bit resolution
  analogSetAttenuation(ADC_11db);        // 0-3.3V range (full scale)
  analogSetWidth(12);                    // 12-bit width
  
  // Set ADC sampling to maximum speed
  //analogSetCycles(8);                    // Faster ADC conversion
  
  // Attach ADC to pin
  //adcAttachPin(MIC_PIN);
  
  // Calculate sample period
  samplePeriod = 1000000L / SAMPLE_RATE;
  
  // Wait for serial
  delay(2000);
  
  // Startup LED sequence (3 fast blinks)
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
  
  // Print configuration
  Serial.println("===========================================");
  Serial.println("ESP32 MAX4466 High-Quality Audio System");
  Serial.println("===========================================");
  Serial.print("Sample Rate: ");
  Serial.print(SAMPLE_RATE);
  Serial.println(" Hz");
  Serial.print("Baud Rate: ");
  Serial.println(BAUD_RATE);
  Serial.print("Buffer Size: ");
  Serial.println(SAMPLES_PER_CHUNK);
  Serial.print("ADC Resolution: ");
  Serial.print(ADC_RESOLUTION);
  Serial.println(" bits");
  Serial.println("===========================================");
  Serial.println("MAX4466 Ready - Adjust gain pot for best signal");
  Serial.println("Commands: START, STOP, TEST, STATS, CALIBRATE");
  Serial.println("===========================================");
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  // Check for commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "START") {
      startRecording();
    } 
    else if (command == "STOP") {
      stopRecording();
    }
    else if (command == "TEST") {
      testMode();
    }
    else if (command == "STATS") {
      printStats();
    }
    else if (command == "CALIBRATE") {
      calibrateADC();
    }
  }
  
  // Record audio if active
  if (isRecording) {
    recordAudioOptimized();
  }
  
  // Print stats every 5 seconds when recording
  if (isRecording && (millis() - lastStatsTime > 5000)) {
    lastStatsTime = millis();
    // Uncomment to debug:
    // printRecordingStats();
  }
}

// ============================================================================
// RECORDING FUNCTIONS - OPTIMIZED FOR MAX4466
// ============================================================================

void startRecording() {
  isRecording = true;
  bufferIndex = 0;
  lastSampleTime = micros();
  samplesProcessed = 0;
  lastStatsTime = millis();
  dcOffset = ADC_CENTER;  // Initialize DC offset
  peakValue = 0;
  
  digitalWrite(LED_PIN, HIGH);
  
  Serial.println("Recording started...");
}

void stopRecording() {
  isRecording = false;
  
  // Send remaining buffer
  if (bufferIndex > 0) {
    sendBuffer();
  }
  
  digitalWrite(LED_PIN, LOW);
  
  Serial.println("Recording stopped.");
  Serial.print("Total samples: ");
  Serial.println(samplesProcessed);
}

void recordAudioOptimized() {
  unsigned long currentTime = micros();
  
  // Precise timing
  if (currentTime - lastSampleTime >= samplePeriod) {
    lastSampleTime = currentTime;
    
    // Read raw ADC value (0-4095)
    int rawValue = analogRead(MIC_PIN);
    
    // Convert to signed 16-bit centered at 0
    // MAX4466 output is already centered at VCC/2 (~1.65V = ADC 2048)
    int32_t sample32 = rawValue - ADC_CENTER;
    
    // Apply DC offset filter (high-pass filter)
    if (ENABLE_DC_FILTER) {
      sample32 = applyDCFilter(sample32);
    }
    
    // Scale to 16-bit range
    // MAX4466 gives better signal, so we can use higher scaling
    // Adjust this multiplier based on your gain pot setting:
    // - Gain pot at min (25x): use multiplier 64
    // - Gain pot at mid (50x): use multiplier 32
    // - Gain pot at max (125x): use multiplier 16
    sample32 *= 32;  // Adjust this based on your pot setting!
    
    // Apply noise gate
    if (ENABLE_NOISE_GATE) {
      if (abs(sample32) < NOISE_THRESHOLD) {
        sample32 = 0;  // Silence below threshold
      }
    }
    
    // Apply automatic gain control (optional)
    if (ENABLE_AGC) {
      sample32 = applyAGC(sample32);
    }
    
    // Clamp to 16-bit range
    int16_t sample = constrain(sample32, -32768, 32767);
    
    // Track peak value for monitoring
    int16_t absSample = abs(sample);
    if (absSample > peakValue) {
      peakValue = absSample;
    }
    
    // Store in buffer
    audioBuffer[bufferIndex++] = sample;
    samplesProcessed++;
    
    // Send buffer when full
    if (bufferIndex >= SAMPLES_PER_CHUNK) {
      sendBuffer();
      bufferIndex = 0;
      
      // Blink LED
      ledState = !ledState;
      digitalWrite(LED_PIN, ledState);
    }
  }
}

void sendBuffer() {
  // Send as binary (2 bytes per sample, little-endian)
  Serial.write((uint8_t*)audioBuffer, bufferIndex * 2);
  Serial.flush();
}

// ============================================================================
// AUDIO PROCESSING FUNCTIONS
// ============================================================================

int32_t applyDCFilter(int32_t sample) {
  // High-pass filter to remove DC offset
  // This removes low-frequency drift and centers signal at 0
  const float alpha = 0.95;  // Filter coefficient (0.95 = ~20Hz cutoff at 16kHz)
  
  dcOffset = alpha * dcOffset + (1.0 - alpha) * sample;
  return sample - (int32_t)dcOffset;
}

int32_t applyAGC(int32_t sample) {
  // Automatic Gain Control
  // Adapts gain based on signal level
  const int32_t targetLevel = 16384;  // Target RMS level (50% of 16-bit range)
  const float attackRate = 0.001;
  const float releaseRate = 0.0001;
  
  int32_t absSample = abs(sample);
  
  if (absSample > agcLevel) {
    // Attack: reduce gain quickly when signal is loud
    agcLevel = agcLevel + (absSample - agcLevel) * attackRate;
  } else {
    // Release: increase gain slowly when signal is quiet
    agcLevel = agcLevel + (absSample - agcLevel) * releaseRate;
  }
  
  // Prevent division by zero
  if (agcLevel < 100) agcLevel = 100;
  
  // Apply gain
  float gain = (float)targetLevel / (float)agcLevel;
  gain = constrain(gain, 0.1, 10.0);  // Limit gain range
  
  return (int32_t)(sample * gain);
}

// ============================================================================
// TEST & CALIBRATION FUNCTIONS
// ============================================================================

void testMode() {
  Serial.println("Test mode - sending 100 samples...");
  digitalWrite(LED_PIN, HIGH);
  
  int16_t testBuffer[100];
  
  for (int i = 0; i < 100; i++) {
    int rawValue = analogRead(MIC_PIN);
    int32_t sample = (rawValue - ADC_CENTER) * 32;
    testBuffer[i] = constrain(sample, -32768, 32767);
    delayMicroseconds(1000000 / SAMPLE_RATE);  // Maintain sample rate
  }
  
  // Send test samples
  Serial.write((uint8_t*)testBuffer, 200);
  Serial.flush();
  
  digitalWrite(LED_PIN, LOW);
  Serial.println("Test complete.");
}

void calibrateADC() {
  Serial.println("===========================================");
  Serial.println("ADC Calibration Mode");
  Serial.println("===========================================");
  Serial.println("Keep microphone in quiet environment...");
  Serial.println("Sampling for 3 seconds...");
  
  digitalWrite(LED_PIN, HIGH);
  
  // Sample for 3 seconds
  int32_t sum = 0;
  int32_t samples = 0;
  int minVal = 4095;
  int maxVal = 0;
  
  unsigned long startTime = millis();
  while (millis() - startTime < 3000) {
    int rawValue = analogRead(MIC_PIN);
    sum += rawValue;
    samples++;
    
    if (rawValue < minVal) minVal = rawValue;
    if (rawValue > maxVal) maxVal = rawValue;
    
    delayMicroseconds(1000);
  }
  
  int avgValue = sum / samples;
  int noiseLevel = maxVal - minVal;
  
  digitalWrite(LED_PIN, LOW);
  
  Serial.println("===========================================");
  Serial.println("Calibration Results:");
  Serial.println("===========================================");
  Serial.print("Average ADC Value: ");
  Serial.print(avgValue);
  Serial.print(" (Expected: ~2048, Diff: ");
  Serial.print(avgValue - 2048);
  Serial.println(")");
  Serial.print("Min Value: ");
  Serial.println(minVal);
  Serial.print("Max Value: ");
  Serial.println(maxVal);
  Serial.print("Noise Level: ");
  Serial.print(noiseLevel);
  Serial.println(" ADC units");
  Serial.print("Samples Taken: ");
  Serial.println(samples);
  
  // Recommendations
  Serial.println("===========================================");
  Serial.println("Recommendations:");
  if (abs(avgValue - 2048) > 100) {
    Serial.println("⚠️  DC offset detected! Check connections.");
  } else {
    Serial.println("✓ DC offset is good");
  }
  
  if (noiseLevel > 200) {
    Serial.println("⚠️  High noise level! Check:");
    Serial.println("   - Ground connection");
    Serial.println("   - Power supply quality");
    Serial.println("   - Distance from noise sources");
  } else if (noiseLevel < 20) {
    Serial.println("⚠️  Very low noise - signal may be too weak");
    Serial.println("   - Turn gain pot clockwise");
  } else {
    Serial.println("✓ Noise level is acceptable");
  }
  
  Serial.println("===========================================");
}

void printStats() {
  Serial.println("===========================================");
  Serial.println("Current Statistics:");
  Serial.println("===========================================");
  Serial.print("Recording: ");
  Serial.println(isRecording ? "YES" : "NO");
  Serial.print("Samples Processed: ");
  Serial.println(samplesProcessed);
  Serial.print("Peak Value: ");
  Serial.print(peakValue);
  Serial.print(" (");
  Serial.print((float)peakValue / 32768.0 * 100.0);
  Serial.println("%)");
  Serial.print("DC Offset: ");
  Serial.println(dcOffset);
  Serial.print("Free Heap: ");
  Serial.print(ESP.getFreeHeap());
  Serial.println(" bytes");
  Serial.println("===========================================");
}

void printRecordingStats() {
  // Internal stats for debugging (optional)
  Serial.print("[STATS] Peak: ");
  Serial.print(peakValue);
  Serial.print(" | Samples: ");
  Serial.print(samplesProcessed);
  Serial.print(" | Heap: ");
  Serial.println(ESP.getFreeHeap());
  
  peakValue = 0;  // Reset peak
}

// ============================================================================
// ADVANCED: DUAL-CORE RECORDING (OPTIONAL)
// ============================================================================
// ESP32 has 2 cores. Use Core 0 for audio, Core 1 for communication
// Uncomment to enable high-performance mode

/*
TaskHandle_t audioTaskHandle = NULL;

void audioTaskCore0(void *parameter) {
  unsigned long lastSample = 0;
  
  while (true) {
    if (isRecording) {
      unsigned long currentTime = micros();
      
      if (currentTime - lastSample >= samplePeriod) {
        lastSample = currentTime;
        
        int rawValue = analogRead(MIC_PIN);
        int32_t sample32 = rawValue - ADC_CENTER;
        
        if (ENABLE_DC_FILTER) {
          sample32 = applyDCFilter(sample32);
        }
        
        sample32 *= 32;
        
        if (ENABLE_NOISE_GATE && abs(sample32) < NOISE_THRESHOLD) {
          sample32 = 0;
        }
        
        int16_t sample = constrain(sample32, -32768, 32767);
        
        audioBuffer[bufferIndex++] = sample;
        samplesProcessed++;
        
        if (bufferIndex >= SAMPLES_PER_CHUNK) {
          sendBuffer();
          bufferIndex = 0;
          ledState = !ledState;
          digitalWrite(LED_PIN, ledState);
        }
      }
    } else {
      vTaskDelay(1);  // Sleep when not recording
    }
  }
}

void startDualCoreRecording() {
  isRecording = true;
  bufferIndex = 0;
  samplesProcessed = 0;
  dcOffset = ADC_CENTER;
  
  digitalWrite(LED_PIN, HIGH);
  
  // Create task on Core 0
  xTaskCreatePinnedToCore(
    audioTaskCore0,
    "AudioTask",
    10000,
    NULL,
    2,  // Higher priority
    &audioTaskHandle,
    0   // Core 0
  );
  
  Serial.println("Dual-core recording started on Core 0");
}

void stopDualCoreRecording() {
  isRecording = false;
  
  if (audioTaskHandle != NULL) {
    vTaskDelete(audioTaskHandle);
    audioTaskHandle = NULL;
  }
  
  if (bufferIndex > 0) {
    sendBuffer();
  }
  
  digitalWrite(LED_PIN, LOW);
  Serial.println("Dual-core recording stopped");
}
*/

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Optional: Spectral analysis (requires FFT library)
// Can be used for real-time frequency analysis

/*
#include "arduinoFFT.h"

arduinoFFT FFT = arduinoFFT();
const uint16_t FFT_SAMPLES = 256;
double vReal[FFT_SAMPLES];
double vImag[FFT_SAMPLES];

void analyzeSpectrum() {
  // Collect samples
  for (int i = 0; i < FFT_SAMPLES; i++) {
    vReal[i] = analogRead(MIC_PIN) - ADC_CENTER;
    vImag[i] = 0;
    delayMicroseconds(62);  // 16kHz sampling
  }
  
  // Perform FFT
  FFT.Windowing(vReal, FFT_SAMPLES, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.Compute(vReal, vImag, FFT_SAMPLES, FFT_FORWARD);
  FFT.ComplexToMagnitude(vReal, vImag, FFT_SAMPLES);
  
  // Find dominant frequency
  double peak = FFT.MajorPeak(vReal, FFT_SAMPLES, SAMPLE_RATE);
  
  Serial.print("Dominant Frequency: ");
  Serial.print(peak);
  Serial.println(" Hz");
}
*/