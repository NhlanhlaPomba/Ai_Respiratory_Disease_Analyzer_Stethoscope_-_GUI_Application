"""
Arduino Audio Interface Module
Handles communication with Arduino KY-027 microphone via Serial
"""

import serial
import serial.tools.list_ports
import numpy as np
import time
from typing import Optional, List, Tuple
import struct


class ArduinoAudioInterface:
    """Interface for reading audio data from Arduino via Serial"""
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 230400):
        """
        Initialize Arduino Audio Interface
        
        Args:
            port: COM port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Serial communication speed (default: 230400 for MAX4466, 115200 for KY-027)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_connection: Optional[serial.Serial] = None
        self.is_connected = False
        
    @staticmethod
    def list_available_ports() -> List[str]:
        """
        List all available serial ports
        
        Returns:
            List of available port names
        """
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    
    def connect(self, port: Optional[str] = None) -> bool:
        """
        Connect to Arduino
        
        Args:
            port: COM port to connect to (uses self.port if None)
            
        Returns:
            True if connection successful, False otherwise
        """
        if port:
            self.port = port
            
        if not self.port:
            raise ValueError("No port specified. Use list_available_ports() to find available ports.")
        
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Wait for Arduino to reset and stabilize
            time.sleep(2)
            
            # Clear any buffered data
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()
            
            self.is_connected = True
            print(f"✓ Connected to Arduino on {self.port}")
            return True
            
        except serial.SerialException as e:
            print(f"✗ Failed to connect to {self.port}: {str(e)}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.is_connected = False
            print("✓ Disconnected from Arduino")
    
    def send_command(self, command: str) -> bool:
        """
        Send command to Arduino
        
        Args:
            command: Command string to send
            
        Returns:
            True if command sent successfully
        """
        if not self.is_connected:
            print("✗ Not connected to Arduino")
            return False
        
        try:
            self.serial_connection.write(f"{command}\n".encode())
            return True
        except Exception as e:
            print(f"✗ Error sending command: {str(e)}")
            return False
    
    def record_audio(self, duration: float = 6.0, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Record audio from Arduino microphone
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Sampling rate (Hz) - must match Arduino code
                        8000 for KY-027, 16000 for MAX4466
            
        Returns:
            Tuple of (audio_data, sample_rate)
            audio_data: numpy array of normalized audio samples (-1.0 to 1.0)
        """
        if not self.is_connected:
            raise Exception("Not connected to Arduino. Call connect() first.")
        
        # Calculate total samples needed
        total_samples = int(duration * sample_rate)
        
        # Clear buffer
        self.serial_connection.reset_input_buffer()
        
        # Send start recording command
        print(f"🎙️ Recording {duration}s from Arduino...")
        self.send_command("START")
        
        # Read audio samples
        audio_samples = []
        start_time = time.time()
        timeout = duration + 5  # Extra time for safety
        
        while len(audio_samples) < total_samples:
            # Check timeout
            if time.time() - start_time > timeout:
                print("⚠️ Recording timeout")
                break
            
            # Read 2 bytes (16-bit sample)
            if self.serial_connection.in_waiting >= 2:
                try:
                    # Read 2 bytes as signed 16-bit integer (little-endian)
                    raw_bytes = self.serial_connection.read(2)
                    sample = struct.unpack('<h', raw_bytes)[0]  # signed short
                    
                    # Normalize to -1.0 to 1.0 range
                    normalized_sample = sample / 32768.0
                    audio_samples.append(normalized_sample)
                    
                except Exception as e:
                    print(f"⚠️ Error reading sample: {str(e)}")
                    continue
        
        # Send stop command
        self.send_command("STOP")
        
        # Convert to numpy array
        audio_data = np.array(audio_samples, dtype=np.float32)
        
        # Ensure correct length (pad or truncate)
        if len(audio_data) < total_samples:
            # Pad with zeros if too short
            padding = total_samples - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant')
            print(f"⚠️ Recording shorter than expected, padded {padding} samples")
        elif len(audio_data) > total_samples:
            # Truncate if too long
            audio_data = audio_data[:total_samples]
        
        print(f"✓ Recorded {len(audio_data)} samples ({len(audio_data)/sample_rate:.2f}s)")
        
        return audio_data, sample_rate
    
    def test_connection(self) -> bool:
        """
        Test connection by reading a few samples
        
        Returns:
            True if test successful
        """
        if not self.is_connected:
            print("✗ Not connected")
            return False
        
        print("Testing Arduino connection...")
        
        try:
            # Clear buffer
            self.serial_connection.reset_input_buffer()
            
            # Send test command
            self.send_command("TEST")
            
            # Try to read a few samples
            test_samples = []
            start_time = time.time()
            
            while len(test_samples) < 10 and time.time() - start_time < 2:
                if self.serial_connection.in_waiting >= 2:
                    raw_bytes = self.serial_connection.read(2)
                    sample = struct.unpack('<h', raw_bytes)[0]
                    test_samples.append(sample)
            
            if len(test_samples) > 0:
                print(f"✓ Connection test passed! Read {len(test_samples)} samples")
                print(f"  Sample values: {test_samples[:5]}")
                return True
            else:
                print("✗ Connection test failed - no data received")
                return False
                
        except Exception as e:
            print(f"✗ Connection test failed: {str(e)}")
            return False
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_arduino_port(description_keywords: List[str] = None) -> Optional[str]:
    """
    Automatically find Arduino port based on description
    
    Args:
        description_keywords: Keywords to search in port description
                            (default: ['Arduino', 'CH340', 'USB Serial'])
    
    Returns:
        Port name if found, None otherwise
    """
    if description_keywords is None:
        description_keywords = ['Arduino', 'CH340', 'USB Serial', 'USB-SERIAL', 'CP210x']
    
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        description = port.description.lower()
        for keyword in description_keywords:
            if keyword.lower() in description:
                print(f"✓ Found Arduino-like device: {port.device} ({port.description})")
                return port.device
    
    print("✗ No Arduino device found automatically")
    return None


def test_arduino_audio(port: str = None, duration: float = 2.0):
    """
    Test function to verify Arduino audio recording
    
    Args:
        port: COM port (auto-detect if None)
        duration: Test recording duration
    """
    print("=" * 60)
    print("Arduino Audio Interface Test")
    print("=" * 60)
    
    # List available ports
    print("\nAvailable ports:")
    ports = ArduinoAudioInterface.list_available_ports()
    for i, p in enumerate(ports, 1):
        print(f"  {i}. {p}")
    
    # Auto-find or use specified port
    if port is None:
        port = find_arduino_port()
        if port is None:
            print("\nPlease specify a port manually.")
            return
    
    # Connect and test
    arduino = ArduinoAudioInterface(port=port, baudrate=115200)
    
    try:
        if arduino.connect():
            print("\n" + "=" * 60)
            print("Connection Test")
            print("=" * 60)
            arduino.test_connection()
            
            print("\n" + "=" * 60)
            print("Recording Test")
            print("=" * 60)
            audio_data, sample_rate = arduino.record_audio(duration=duration, sample_rate=8000)
            
            print(f"\nRecorded Audio Statistics:")
            print(f"  Duration: {len(audio_data)/sample_rate:.2f}s")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Total samples: {len(audio_data)}")
            print(f"  Min value: {np.min(audio_data):.4f}")
            print(f"  Max value: {np.max(audio_data):.4f}")
            print(f"  Mean value: {np.mean(audio_data):.4f}")
            print(f"  RMS: {np.sqrt(np.mean(audio_data**2)):.4f}")
            
            print("\n✓ Test completed successfully!")
            
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        arduino.disconnect()


if __name__ == "__main__":
    # Run test when module is executed directly
    print("Running ESP32 Audio Interface Test...")
    print("Make sure your ESP32 is connected and programmed.\n")
    test_arduino_audio(duration=2.0)