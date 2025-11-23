"""
Advanced Respiratory Disease Classification Desktop Application - Dark Theme
Features:
- Arduino microphone integration via Serial
- Upload and analyze WAV files
- Multi-disease classification with visual cards
- Confidence visualization
- Export results to JSON/text
- Audio playback and waveform + spectrogram visualization
"""

import os
import sys
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, filtfilt
import tensorflow as tf
from datetime import datetime
import json
import traceback

# GUI imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading

# Import Arduino audio interface
try:
    from arduino_audio import ArduinoAudioInterface, find_arduino_port
    ARDUINO_AVAILABLE = True
except ImportError:
    ARDUINO_AVAILABLE = False
    print("⚠️  arduino_audio.py not found. Arduino features will be disabled.")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    SAMPLE_RATE = 22050
    ARDUINO_SAMPLE_RATE = 16000  # MAX4466 sampling rate (16kHz for high quality)
    DURATION = 6  # seconds
    N_MELS = 128
    N_MFCC = 13
    N_CHROMA = 12
    MAX_PAD_LEN = 259
    MODEL_PATH = 'advanced_respiratory_model_quant.tflite'
    DISEASE_CLASSES = [
        'Bronchiectasis',
        'Bronchiolitis',
        'COPD',
        'Healthy',
        'Pneumonia',
        'URTI'
    ]
    # Dark theme colors
    COLORS = {
        'bg_dark': '#1a1a2e',
        'bg_medium': '#16213e',
        'bg_light': '#0f3460',
        'accent': '#00d4ff',
        'accent_hover': '#00a8cc',
        'success': '#2ecc71',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'text_light': '#ecf0f1',
        'text_muted': '#95a5a6',
        'card_bg': '#1e1e3f',
        'card_border': '#2d2d5f',
        'disabled': '#34495e'
    }
    
    DISEASE_ICONS = {
        'Bronchiectasis': '🫁',
        'Bronchiolitis': '🌡️',
        'COPD': '💨',
        'Healthy': '✅',
        'Pneumonia': '🦠',
        'URTI': '🤧'
    }

# ============================================================================
# AUDIO PROCESSING
# ============================================================================

class AudioProcessor:
    """Handle all audio processing operations"""

    @staticmethod
    def preprocess_audio(audio, sr):
        """Enhanced preprocessing for respiratory sounds"""
        audio = np.asarray(audio, dtype=np.float32).flatten()

        if audio.size == 0:
            return np.zeros(int(Config.SAMPLE_RATE * Config.DURATION), dtype=np.float32)

        try:
            audio = librosa.effects.preemphasis(audio, coef=0.97)
        except Exception:
            audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        try:
            rms = np.mean(librosa.feature.rms(y=audio, frame_length=2048))
            audio = audio / (rms + 1e-8)
        except Exception:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)

        audio = np.clip(audio, -1.0, 1.0)

        try:
            nyquist = sr / 2.0
            def norm(low, high):
                return [max(low / nyquist, 1e-6), min(high / nyquist, 0.9999)]

            b1, a1 = butter(4, norm(50, 200), btype='band')
            low_freq = filtfilt(b1, a1, audio)

            b2, a2 = butter(4, norm(200, 1000), btype='band')
            mid_freq = filtfilt(b2, a2, audio)

            b3, a3 = butter(4, norm(1000, min(4000, nyquist - 1)), btype='band')
            high_freq = filtfilt(b3, a3, audio)

            audio = 0.3 * low_freq + 0.4 * mid_freq + 0.3 * high_freq
        except Exception:
            pass

        return audio

    @staticmethod
    def extract_features(audio, sr):
        """Extract multi-channel features"""
        if sr != Config.SAMPLE_RATE:
            try:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=Config.SAMPLE_RATE)
                sr = Config.SAMPLE_RATE
            except Exception:
                pass

        expected_len = int(Config.SAMPLE_RATE * Config.DURATION)
        if len(audio) < expected_len:
            audio = np.pad(audio, (0, expected_len - len(audio)))
        else:
            audio = audio[:expected_len]

        audio = AudioProcessor.preprocess_audio(audio, sr)

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=Config.N_MELS,
            hop_length=512, win_length=1024, power=2.0
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=Config.N_MFCC, hop_length=512
        )

        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr, hop_length=512
        )

        def resize_feature(feature, target_time):
            if feature.shape[1] > target_time:
                return feature[:, :target_time]
            elif feature.shape[1] < target_time:
                pad_width = target_time - feature.shape[1]
                return np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
            else:
                return feature

        mel_db = resize_feature(mel_db, Config.MAX_PAD_LEN)
        mfcc = resize_feature(mfcc, Config.MAX_PAD_LEN)
        chroma = resize_feature(chroma, Config.MAX_PAD_LEN)

        combined_features = np.vstack([mel_db, mfcc, chroma])

        assert combined_features.shape == (Config.N_MELS + Config.N_MFCC + Config.N_CHROMA, Config.MAX_PAD_LEN), \
            f"Features shape mismatch: got {combined_features.shape}"

        return combined_features

    @staticmethod
    def load_audio_file(file_path):
        """Load audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE, duration=Config.DURATION, mono=True)
            expected_len = int(Config.SAMPLE_RATE * Config.DURATION)
            if len(audio) < expected_len:
                audio = np.pad(audio, (0, expected_len - len(audio)))
            return audio, sr
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")

# ============================================================================
# MODEL INFERENCE
# ============================================================================

class ModelInference:
    """Handle TensorFlow Lite model inference"""

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            raise Exception(f"Failed to load TFLite model: {e}")

        self.input_shape = tuple(self.input_details[0]['shape'])
        print("TFLite model loaded.")

    def predict(self, features_2d):
        input_data = features_2d.astype(np.float32)
        input_data = input_data.reshape((1, input_data.shape[0], input_data.shape[1], 1)).astype(np.float32)

        if self.input_details[0]['dtype'] == np.uint8 or self.input_details[0]['dtype'] == np.int8:
            scale, zero_point = self.input_details[0]['quantization']
            if scale == 0:
                q_input = np.clip(np.round(input_data), np.iinfo(self.input_details[0]['dtype']).min,
                                  np.iinfo(self.input_details[0]['dtype']).max).astype(self.input_details[0]['dtype'])
            else:
                q = np.round(input_data / scale + zero_point).astype(np.int32)
                qmin = np.iinfo(self.input_details[0]['dtype']).min
                qmax = np.iinfo(self.input_details[0]['dtype']).max
                q = np.clip(q, qmin, qmax).astype(self.input_details[0]['dtype'])
                q_input = q
            self.interpreter.set_tensor(self.input_details[0]['index'], q_input)
        else:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index']).squeeze()

        if self.output_details[0]['dtype'] == np.uint8 or self.output_details[0]['dtype'] == np.int8:
            scale, zero_point = self.output_details[0]['quantization']
            if scale is None:
                scale = 1.0
                zero_point = 0
            output_data = scale * (output_data.astype(np.float32) - zero_point)

        probs = self._softmax(output_data)
        return probs

    @staticmethod
    def _softmax(x):
        x = np.asarray(x, dtype=np.float32)
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum() + 1e-12)

# ============================================================================
# CUSTOM WIDGETS
# ============================================================================

class ModernButton(tk.Canvas):
    """Custom modern button with hover effects"""
    
    def __init__(self, parent, text, command, bg_color, hover_color, fg_color='white', width=200, height=50, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent['bg'], 
                        highlightthickness=0, cursor='hand2', **kwargs)
        
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.fg_color = fg_color
        self.text = text
        self.width = width
        self.height = height
        self.is_hovered = False
        self.is_disabled = False
        
        self.draw_button()
        self.bind('<Button-1>', self.on_click)
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
    
    def draw_button(self):
        self.delete('all')
        
        if self.is_disabled:
            color = Config.COLORS['disabled']
            text_color = Config.COLORS['text_muted']
            self.config(cursor='arrow')
        else:
            color = self.hover_color if self.is_hovered else self.bg_color
            text_color = self.fg_color
            self.config(cursor='hand2')
        
        r = 10
        self.create_arc((0, 0, 2*r, 2*r), start=90, extent=90, fill=color, outline=color)
        self.create_arc((self.width-2*r, 0, self.width, 2*r), start=0, extent=90, fill=color, outline=color)
        self.create_arc((0, self.height-2*r, 2*r, self.height), start=180, extent=90, fill=color, outline=color)
        self.create_arc((self.width-2*r, self.height-2*r, self.width, self.height), start=270, extent=90, fill=color, outline=color)
        
        self.create_rectangle((r, 0, self.width-r, self.height), fill=color, outline=color)
        self.create_rectangle((0, r, self.width, self.height-r), fill=color, outline=color)
        
        self.create_text(self.width/2, self.height/2, text=self.text, 
                        fill=text_color, font=('Helvetica', 11, 'bold'))
    
    def on_enter(self, e):
        if not self.is_disabled:
            self.is_hovered = True
            self.draw_button()
    
    def on_leave(self, e):
        self.is_hovered = False
        self.draw_button()
    
    def on_click(self, e):
        if not self.is_disabled and self.command:
            self.command()
    
    def set_state(self, state):
        """Set button state: tk.NORMAL or tk.DISABLED"""
        self.is_disabled = (state == tk.DISABLED)
        self.draw_button()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class RespiratoryDiseaseApp:
    """Main application class"""

    def __init__(self, root):
        self.root = root
        self.root.title("Respiratory Disease AI Classifier")
        self.root.geometry("1400x900")
        self.root.configure(bg=Config.COLORS['bg_dark'])

        # Variables
        self.model = None
        self.model_loaded = False
        self.current_audio = None
        self.current_sr = None
        self.results_history = []
        self.arduino = None
        self.arduino_connected = False

        plt.style.use('dark_background')

        # Load model
        try:
            if os.path.exists(Config.MODEL_PATH):
                self.model = ModelInference(Config.MODEL_PATH)
                self.model_loaded = True
            else:
                self.model_loaded = False
        except Exception as e:
            self.model_loaded = False

        self.setup_ui()
        self.update_status("Ready • Upload file or connect Arduino")

    def setup_ui(self):
        """Setup UI"""
        # Header
        header = tk.Frame(self.root, bg=Config.COLORS['bg_medium'], height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="🫁 Respiratory Disease AI Classifier",
                font=("Helvetica", 26, "bold"),
                bg=Config.COLORS['bg_medium'],
                fg=Config.COLORS['text_light']).pack(side=tk.LEFT, padx=30, pady=20)

        tk.Label(header, text="Advanced ML-Powered Respiratory Analysis",
                font=("Helvetica", 11),
                bg=Config.COLORS['bg_medium'],
                fg=Config.COLORS['text_muted']).pack(side=tk.LEFT, padx=10)

        # Main container
        main_container = tk.Frame(self.root, bg=Config.COLORS['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left panel
        left_panel = tk.Frame(main_container, bg=Config.COLORS['bg_medium'], width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)
        self.setup_control_panel(left_panel)

        # Right panel
        right_panel = tk.Frame(main_container, bg=Config.COLORS['bg_dark'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.setup_results_panel(right_panel)

        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready",
                                   bd=0, anchor=tk.W,
                                   bg=Config.COLORS['bg_light'],
                                   fg=Config.COLORS['text_muted'],
                                   font=("Helvetica", 10),
                                   padx=20, pady=10)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_control_panel(self, parent):
        """Setup control panel"""
        canvas = tk.Canvas(parent, bg=Config.COLORS['bg_medium'], highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=Config.COLORS['bg_medium'])

        scrollable_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', bind_mousewheel)
        canvas.bind('<Leave>', unbind_mousewheel)

        tk.Label(scrollable_frame, text="Control Panel",
                font=("Helvetica", 18, "bold"),
                bg=Config.COLORS['bg_medium'],
                fg=Config.COLORS['text_light']).pack(pady=20, padx=20)

        # Arduino Section
        if ARDUINO_AVAILABLE:
            self._create_section(scrollable_frame, "🎤 Arduino Microphone")
            
            arduino_frame = tk.Frame(scrollable_frame, bg=Config.COLORS['bg_medium'])
            arduino_frame.pack(pady=10, padx=20, fill=tk.X)
            
            self.arduino_status = tk.Label(arduino_frame, text="● Disconnected",
                                          bg=Config.COLORS['bg_medium'],
                                          fg=Config.COLORS['danger'],
                                          font=("Helvetica", 10))
            self.arduino_status.pack(anchor=tk.W, pady=5)
            
            btn_frame_ard = tk.Frame(scrollable_frame, bg=Config.COLORS['bg_medium'])
            btn_frame_ard.pack(pady=10, padx=20, fill=tk.X)
            
            self.connect_arduino_btn = ModernButton(btn_frame_ard, "Connect Arduino", 
                                                    self.connect_arduino_dialog,
                                                    Config.COLORS['accent'], 
                                                    Config.COLORS['accent_hover'],
                                                    width=290, height=45)
            self.connect_arduino_btn.pack()
            
            btn_frame_rec = tk.Frame(scrollable_frame, bg=Config.COLORS['bg_medium'])
            btn_frame_rec.pack(pady=10, padx=20, fill=tk.X)
            
            self.record_arduino_btn = ModernButton(btn_frame_rec, "🎙️ Record from Arduino", 
                                                   self.record_from_arduino,
                                                   Config.COLORS['danger'], '#c0392b',
                                                   width=290, height=45)
            self.record_arduino_btn.pack()
            self.record_arduino_btn.set_state(tk.DISABLED)
        
        # Upload Section
        self._create_section(scrollable_frame, "📁 Audio File")
        
        btn_frame = tk.Frame(scrollable_frame, bg=Config.COLORS['bg_medium'])
        btn_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.upload_btn = ModernButton(btn_frame, "Upload WAV File", self.upload_file,
                                      Config.COLORS['accent'], Config.COLORS['accent_hover'],
                                      width=290, height=45)
        self.upload_btn.pack()

        # Analysis Section
        self._create_section(scrollable_frame, "🔬 Analysis")
        
        btn_frame3 = tk.Frame(scrollable_frame, bg=Config.COLORS['bg_medium'])
        btn_frame3.pack(pady=10, padx=20, fill=tk.X)
        
        self.analyze_btn = ModernButton(btn_frame3, "Analyze Audio", self.analyze_audio,
                                       Config.COLORS['success'], '#27ae60',
                                       width=290, height=45)
        self.analyze_btn.pack()
        self.analyze_btn.set_state(tk.DISABLED)

        # Playback Section
        self._create_section(scrollable_frame, "▶️ Playback")
        
        btn_frame4 = tk.Frame(scrollable_frame, bg=Config.COLORS['bg_medium'])
        btn_frame4.pack(pady=10, padx=20, fill=tk.X)
        
        self.play_btn = ModernButton(btn_frame4, "Play Audio", self.play_audio,
                                    Config.COLORS['bg_light'], Config.COLORS['card_border'],
                                    width=290, height=40)
        self.play_btn.pack()
        self.play_btn.set_state(tk.DISABLED)

        # Export Section
        self._create_section(scrollable_frame, "💾 Export")
        
        btn_frame5 = tk.Frame(scrollable_frame, bg=Config.COLORS['bg_medium'])
        btn_frame5.pack(pady=10, padx=20, fill=tk.X)
        
        self.export_btn = ModernButton(btn_frame5, "Export Results", self.export_results,
                                      Config.COLORS['warning'], '#d68910',
                                      width=290, height=40)
        self.export_btn.pack()
        self.export_btn.set_state(tk.DISABLED)

        # Clear Button
        btn_frame6 = tk.Frame(scrollable_frame, bg=Config.COLORS['bg_medium'])
        btn_frame6.pack(pady=10, padx=20, fill=tk.X)
        
        ModernButton(btn_frame6, "🗑️ Clear All", self.clear_all,
                    Config.COLORS['bg_light'], Config.COLORS['card_border'],
                    width=290, height=35).pack()

        # Info Section
        self._create_section(scrollable_frame, "ℹ️ Disease Classes")
        
        info_frame = tk.Frame(scrollable_frame, bg=Config.COLORS['bg_medium'])
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        for disease in Config.DISEASE_CLASSES:
            icon = Config.DISEASE_ICONS.get(disease, '•')
            tk.Label(info_frame, text=f"{icon} {disease}",
                    bg=Config.COLORS['bg_medium'],
                    fg=Config.COLORS['text_muted'],
                    font=("Helvetica", 9),
                    anchor=tk.W).pack(anchor=tk.W, pady=2)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _create_section(self, parent, title):
        """Create section divider"""
        tk.Frame(parent, height=1, bg=Config.COLORS['card_border']).pack(fill=tk.X, padx=20, pady=15)
        tk.Label(parent, text=title,
                font=("Helvetica", 12, "bold"),
                bg=Config.COLORS['bg_medium'],
                fg=Config.COLORS['accent']).pack(anchor=tk.W, padx=20, pady=5)

    def setup_results_panel(self, parent):
        """Setup results panel"""
        style = ttk.Style()
        style.theme_use('default')
        style.configure('Dark.TNotebook', background=Config.COLORS['bg_dark'], borderwidth=0)
        style.configure('Dark.TNotebook.Tab', background=Config.COLORS['bg_medium'],
                       foreground=Config.COLORS['text_light'], padding=[20, 10],
                       font=('Helvetica', 10, 'bold'))
        style.map('Dark.TNotebook.Tab', background=[('selected', Config.COLORS['bg_light'])],
                 foreground=[('selected', Config.COLORS['accent'])])

        self.notebook = ttk.Notebook(parent, style='Dark.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Results Tab
        results_tab = tk.Frame(self.notebook, bg=Config.COLORS['bg_dark'])
        self.notebook.add(results_tab, text="📊 Results")
        
        results_canvas = tk.Canvas(results_tab, bg=Config.COLORS['bg_dark'], highlightthickness=0)
        results_scrollbar = tk.Scrollbar(results_tab, orient="vertical", command=results_canvas.yview)
        self.results_frame = tk.Frame(results_canvas, bg=Config.COLORS['bg_dark'])
        
        self.results_frame.bind("<Configure>",
            lambda e: results_canvas.configure(scrollregion=results_canvas.bbox("all")))
        
        results_canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        results_canvas.configure(yscrollcommand=results_scrollbar.set)
        
        # Enable mousewheel scrolling for results
        def on_results_mousewheel(event):
            results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_results_mousewheel(event):
            results_canvas.bind_all("<MouseWheel>", on_results_mousewheel)
        
        def unbind_results_mousewheel(event):
            results_canvas.unbind_all("<MouseWheel>")
        
        results_canvas.bind('<Enter>', bind_results_mousewheel)
        results_canvas.bind('<Leave>', unbind_results_mousewheel)
        
        results_canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        results_scrollbar.pack(side="right", fill="y")

        # Visualization Tab
        viz_tab = tk.Frame(self.notebook, bg=Config.COLORS['bg_dark'])
        self.notebook.add(viz_tab, text="📈 Confidence Chart")

        self.fig = Figure(figsize=(10, 6), dpi=100, facecolor=Config.COLORS['bg_dark'])
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Waveform Tab
        waveform_tab = tk.Frame(self.notebook, bg=Config.COLORS['bg_dark'])
        self.notebook.add(waveform_tab, text="🌊 Audio Waveform")

        self.waveform_fig = Figure(figsize=(10, 6), dpi=100, facecolor=Config.COLORS['bg_dark'])
        self.waveform_canvas = FigureCanvasTkAgg(self.waveform_fig, master=waveform_tab)
        self.waveform_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # ========================================================================
    # ARDUINO FUNCTIONS
    # ========================================================================

    def connect_arduino_dialog(self):
        """Show dialog to connect to Arduino"""
        if not ARDUINO_AVAILABLE:
            messagebox.showerror("Arduino Module Missing", 
                               "arduino_audio.py module not found!\nPlease place it in the same directory.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Connect to Arduino")
        dialog.geometry("400x300")
        dialog.configure(bg=Config.COLORS['bg_dark'])
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Select Arduino Port",
                font=("Helvetica", 14, "bold"),
                bg=Config.COLORS['bg_dark'],
                fg=Config.COLORS['text_light']).pack(pady=20)

        # List available ports
        ports = ArduinoAudioInterface.list_available_ports()
        
        if not ports:
            tk.Label(dialog, text="No serial ports found!",
                    bg=Config.COLORS['bg_dark'],
                    fg=Config.COLORS['danger']).pack(pady=10)
            tk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
            return

        port_var = tk.StringVar(value=ports[0])
        
        for port in ports:
            tk.Radiobutton(dialog, text=port, variable=port_var, value=port,
                          bg=Config.COLORS['bg_dark'],
                          fg=Config.COLORS['text_light'],
                          selectcolor=Config.COLORS['bg_light'],
                          font=("Helvetica", 10)).pack(anchor=tk.W, padx=40, pady=5)

        def connect():
            selected_port = port_var.get()
            dialog.destroy()
            self.connect_to_arduino(selected_port)

        tk.Button(dialog, text="Connect", command=connect,
                 bg=Config.COLORS['success'], fg='white',
                 font=("Helvetica", 11, "bold"),
                 padx=30, pady=10).pack(pady=20)

    def connect_to_arduino(self, port):
        """Connect to Arduino on specified port"""
        try:
            self.update_status(f"Connecting to Arduino on {port}...")
            
            self.arduino = ArduinoAudioInterface(port=port, baudrate=230400)  # MAX4466 baud rate
            
            if self.arduino.connect():
                self.arduino_connected = True
                self.arduino_status.config(text=f"● Connected: {port}", 
                                          fg=Config.COLORS['success'])
                self.connect_arduino_btn.text = "Disconnect Arduino"
                self.connect_arduino_btn.command = self.disconnect_arduino
                self.connect_arduino_btn.draw_button()
                self.record_arduino_btn.set_state(tk.NORMAL)
                
                self.update_status(f"✓ Connected to Arduino on {port}")
                messagebox.showinfo("Success", f"Connected to Arduino on {port}")
            else:
                raise Exception("Connection failed")
                
        except Exception as e:
            self.arduino_connected = False
            messagebox.showerror("Connection Error", f"Failed to connect to Arduino:\n{str(e)}")
            self.update_status("✗ Arduino connection failed")

    def disconnect_arduino(self):
        """Disconnect from Arduino"""
        if self.arduino:
            self.arduino.disconnect()
            self.arduino = None
            self.arduino_connected = False
            self.arduino_status.config(text="● Disconnected", 
                                      fg=Config.COLORS['danger'])
            self.connect_arduino_btn.text = "Connect Arduino"
            self.connect_arduino_btn.command = self.connect_arduino_dialog
            self.connect_arduino_btn.draw_button()
            self.record_arduino_btn.set_state(tk.DISABLED)
            self.update_status("✓ Arduino disconnected")

    def record_from_arduino(self):
        """Record audio from Arduino"""
        if not self.arduino_connected or not self.arduino:
            messagebox.showwarning("Not Connected", "Please connect to Arduino first!")
            return

        # Run in thread to avoid blocking UI
        threading.Thread(target=self.arduino_record_thread, daemon=True).start()

    def arduino_record_thread(self):
        """Arduino recording thread"""
        try:
            self.root.after(0, lambda: self.update_status("🎙️ Recording from Arduino..."))
            self.root.after(0, lambda: self.record_arduino_btn.set_state(tk.DISABLED))
            
            # Record from Arduino
            audio_data, sample_rate = self.arduino.record_audio(
                duration=Config.DURATION,
                sample_rate=Config.ARDUINO_SAMPLE_RATE
            )
            
            # Store audio
            self.current_audio = audio_data
            self.current_sr = sample_rate
            
            # Update UI in main thread
            self.root.after(0, self.arduino_record_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self.arduino_record_error(str(e)))

    def arduino_record_complete(self):
        """Handle Arduino recording completion"""
        self.record_arduino_btn.set_state(tk.NORMAL)
        self.analyze_btn.set_state(tk.NORMAL)
        self.play_btn.set_state(tk.NORMAL)
        
        self.display_waveform()
        self.update_status("✓ Arduino recording completed")
        messagebox.showinfo("Success", "Recording from Arduino completed!")

    def arduino_record_error(self, error_msg):
        """Handle Arduino recording error"""
        self.record_arduino_btn.set_state(tk.NORMAL)
        self.update_status("✗ Arduino recording failed")
        messagebox.showerror("Recording Error", f"Failed to record from Arduino:\n{error_msg}")

    # ========================================================================
    # FILE FUNCTIONS
    # ========================================================================

    def upload_file(self):
        """Handle file upload"""
        file_path = filedialog.askopenfilename(
            title="Select WAV Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.update_status(f"Loading: {os.path.basename(file_path)}")
                self.current_audio, self.current_sr = AudioProcessor.load_audio_file(file_path)
                self.update_status(f"✓ Loaded: {os.path.basename(file_path)}")

                self.analyze_btn.set_state(tk.NORMAL)
                self.play_btn.set_state(tk.NORMAL)

                self.display_waveform()
                messagebox.showinfo("Success", "Audio file loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load audio file:\n{str(e)}")
                self.update_status("✗ Error loading file")

    # ========================================================================
    # ANALYSIS FUNCTIONS
    # ========================================================================

    def analyze_audio(self):
        """Analyze current audio"""
        if self.current_audio is None:
            messagebox.showwarning("No Audio", "Please load or record audio first!")
            return

        if not self.model_loaded:
            messagebox.showerror("Model Not Loaded", 
                               "Please place a valid .tflite model and restart.")
            return

        try:
            self.update_status("🔬 Extracting features...")
            features = AudioProcessor.extract_features(self.current_audio, self.current_sr)

            self.update_status("🧠 Running AI inference...")
            probabilities = self.model.predict(features)

            predicted_idx = int(np.argmax(probabilities))
            predicted_disease = Config.DISEASE_CLASSES[predicted_idx]
            confidence = float(probabilities[predicted_idx] * 100.0)

            result = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'predicted_disease': predicted_disease,
                'confidence': confidence,
                'all_probabilities': {disease: float(prob * 100.0)
                                     for disease, prob in zip(Config.DISEASE_CLASSES, probabilities)}
            }
            self.results_history.append(result)

            self.display_results(result)
            self.visualize_predictions(probabilities)

            self.export_btn.set_state(tk.NORMAL)
            self.update_status(f"✓ Analysis complete: {predicted_disease} ({confidence:.1f}%)")

        except Exception as e:
            tb = traceback.format_exc()
            messagebox.showerror("Analysis Error", f"Failed to analyze audio:\n{str(e)}\n\n{tb}")
            self.update_status("✗ Analysis failed")

    def display_results(self, result):
        """Display results with modern cards"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Header Card
        header_card = tk.Frame(self.results_frame, bg=Config.COLORS['card_bg'], relief=tk.FLAT, bd=2)
        header_card.pack(fill=tk.X, padx=20, pady=(20, 10))

        tk.Label(header_card, text="Analysis Complete",
                font=("Helvetica", 20, "bold"),
                bg=Config.COLORS['card_bg'],
                fg=Config.COLORS['text_light']).pack(anchor=tk.W, padx=20, pady=(20, 5))

        tk.Label(header_card, text=f"📅 {result['timestamp']}",
                font=("Helvetica", 10),
                bg=Config.COLORS['card_bg'],
                fg=Config.COLORS['text_muted']).pack(anchor=tk.W, padx=20, pady=(0, 20))

        # Primary Diagnosis Card
        primary_card = tk.Frame(self.results_frame, bg=Config.COLORS['success'], relief=tk.FLAT, bd=2)
        primary_card.pack(fill=tk.X, padx=20, pady=10)

        icon = Config.DISEASE_ICONS.get(result['predicted_disease'], '🔍')
        
        tk.Label(primary_card, text=f"{icon} Primary Diagnosis",
                font=("Helvetica", 12, "bold"),
                bg=Config.COLORS['success'],
                fg='white').pack(anchor=tk.W, padx=20, pady=(15, 5))

        tk.Label(primary_card, text=result['predicted_disease'],
                font=("Helvetica", 28, "bold"),
                bg=Config.COLORS['success'],
                fg='white').pack(anchor=tk.W, padx=20, pady=5)

        tk.Label(primary_card, text=f"Confidence: {result['confidence']:.1f}%",
                font=("Helvetica", 16),
                bg=Config.COLORS['success'],
                fg='white').pack(anchor=tk.W, padx=20, pady=(0, 15))

        # Confidence Interpretation
        if result['confidence'] > 80:
            interpretation = "✅ HIGH CONFIDENCE"
            interp_color = Config.COLORS['success']
        elif result['confidence'] > 60:
            interpretation = "⚠️ MODERATE CONFIDENCE"
            interp_color = Config.COLORS['warning']
        else:
            interpretation = "⚡ LOW CONFIDENCE"
            interp_color = Config.COLORS['danger']

        interp_card = tk.Frame(self.results_frame, bg=interp_color, relief=tk.FLAT, bd=2)
        interp_card.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(interp_card, text=interpretation,
                font=("Helvetica", 14, "bold"),
                bg=interp_color,
                fg='white').pack(padx=20, pady=15)

        # All Probabilities Card
        probs_card = tk.Frame(self.results_frame, bg=Config.COLORS['card_bg'], relief=tk.FLAT, bd=2)
        probs_card.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(probs_card, text="📊 All Disease Probabilities",
                font=("Helvetica", 14, "bold"),
                bg=Config.COLORS['card_bg'],
                fg=Config.COLORS['text_light']).pack(anchor=tk.W, padx=20, pady=(20, 15))

        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)

        for disease, prob in sorted_probs:
            disease_row = tk.Frame(probs_card, bg=Config.COLORS['card_bg'])
            disease_row.pack(fill=tk.X, padx=20, pady=8)

            icon = Config.DISEASE_ICONS.get(disease, '•')
            
            label_frame = tk.Frame(disease_row, bg=Config.COLORS['card_bg'])
            label_frame.pack(side=tk.LEFT, fill=tk.Y)
            
            tk.Label(label_frame, text=f"{icon} {disease}",
                    font=("Helvetica", 11),
                    bg=Config.COLORS['card_bg'],
                    fg=Config.COLORS['text_light'],
                    width=20, anchor=tk.W).pack()

            bar_frame = tk.Frame(disease_row, bg=Config.COLORS['bg_dark'], height=25, relief=tk.FLAT)
            bar_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

            fill_width = int((prob / 100) * 300)
            if prob == max(result['all_probabilities'].values()):
                bar_color = Config.COLORS['success']
            elif prob > 20:
                bar_color = Config.COLORS['accent']
            else:
                bar_color = Config.COLORS['bg_light']

            tk.Frame(bar_frame, bg=bar_color, width=fill_width, height=25).pack(side=tk.LEFT)

            tk.Label(disease_row, text=f"{prob:.1f}%",
                    font=("Helvetica", 11, "bold"),
                    bg=Config.COLORS['card_bg'],
                    fg=Config.COLORS['accent'],
                    width=8).pack(side=tk.RIGHT)

        tk.Frame(probs_card, height=20, bg=Config.COLORS['card_bg']).pack()

        # Disclaimer Card
        disclaimer_card = tk.Frame(self.results_frame, bg=Config.COLORS['bg_light'], relief=tk.FLAT, bd=2)
        disclaimer_card.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(disclaimer_card, text="⚠️ Medical Disclaimer",
                font=("Helvetica", 12, "bold"),
                bg=Config.COLORS['bg_light'],
                fg=Config.COLORS['warning']).pack(anchor=tk.W, padx=20, pady=(15, 5))

        disclaimer_text = ("This is an AI-assisted diagnostic tool for educational purposes. "
                          "Results should be verified by qualified healthcare professionals. "
                          "Do not use as the sole basis for medical decisions.")
        
        tk.Label(disclaimer_card, text=disclaimer_text,
                font=("Helvetica", 9),
                bg=Config.COLORS['bg_light'],
                fg=Config.COLORS['text_muted'],
                wraplength=600, justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 15))

    def visualize_predictions(self, probabilities):
        """Visualize predictions"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(Config.COLORS['bg_dark'])

        probs_pct = np.array(probabilities) * 100.0
        colors = [Config.COLORS['success'] if p == probs_pct.max() 
                 else Config.COLORS['accent'] for p in probs_pct]

        bars = ax.bar(range(len(Config.DISEASE_CLASSES)), probs_pct, color=colors,
                     edgecolor=Config.COLORS['text_muted'], linewidth=1.5, alpha=0.9)

        ax.set_xlabel('Disease', fontsize=13, fontweight='bold', color=Config.COLORS['text_light'])
        ax.set_ylabel('Probability (%)', fontsize=13, fontweight='bold', color=Config.COLORS['text_light'])
        ax.set_title('Disease Classification Confidence', fontsize=16, fontweight='bold',
                    color=Config.COLORS['text_light'], pad=20)
        ax.set_xticks(range(len(Config.DISEASE_CLASSES)))
        ax.set_xticklabels(Config.DISEASE_CLASSES, rotation=45, ha='right', color=Config.COLORS['text_light'])
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.2, linestyle='--', color=Config.COLORS['text_muted'])
        ax.tick_params(colors=Config.COLORS['text_light'])

        for spine in ax.spines.values():
            spine.set_edgecolor(Config.COLORS['text_muted'])

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 2.0, f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold',
                   color=Config.COLORS['text_light'], fontsize=11)

        self.fig.tight_layout()
        self.canvas.draw()

    def display_waveform(self):
        """Display waveform and spectrogram"""
        if self.current_audio is None:
            return

        self.waveform_fig.clear()

        time = np.linspace(0, len(self.current_audio) / self.current_sr, len(self.current_audio))

        ax1 = self.waveform_fig.add_subplot(211)
        ax2 = self.waveform_fig.add_subplot(212)
        
        for ax in [ax1, ax2]:
            ax.set_facecolor(Config.COLORS['bg_dark'])

        ax1.plot(time, self.current_audio, linewidth=0.8, color=Config.COLORS['accent'])
        ax1.set_xlabel('Time (s)', fontsize=11, color=Config.COLORS['text_light'])
        ax1.set_ylabel('Amplitude', fontsize=11, color=Config.COLORS['text_light'])
        ax1.set_title('Audio Waveform', fontsize=13, fontweight='bold', color=Config.COLORS['text_light'])
        ax1.grid(True, alpha=0.2, color=Config.COLORS['text_muted'])
        ax1.tick_params(colors=Config.COLORS['text_light'])
        
        for spine in ax1.spines.values():
            spine.set_edgecolor(Config.COLORS['text_muted'])

        try:
            mel_spec = librosa.feature.melspectrogram(y=self.current_audio, sr=self.current_sr,
                                                     n_mels=Config.N_MELS, hop_length=512)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            img = ax2.imshow(mel_db, aspect='auto', origin='lower', cmap='viridis')
            ax2.set_ylabel('Mel Bins', fontsize=11, color=Config.COLORS['text_light'])
            ax2.set_xlabel('Time Frame', fontsize=11, color=Config.COLORS['text_light'])
            ax2.set_title('Mel Spectrogram (dB)', fontsize=13, fontweight='bold',
                         color=Config.COLORS['text_light'])
            ax2.tick_params(colors=Config.COLORS['text_light'])
            
            for spine in ax2.spines.values():
                spine.set_edgecolor(Config.COLORS['text_muted'])
            
            cbar = self.waveform_fig.colorbar(img, ax=ax2, format='%+2.0f dB')
            cbar.ax.yaxis.set_tick_params(color=Config.COLORS['text_light'])
            cbar.outline.set_edgecolor(Config.COLORS['text_muted'])
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=Config.COLORS['text_light'])
        except Exception:
            ax2.text(0.5, 0.5, "Spectrogram unavailable", ha='center', va='center',
                    color=Config.COLORS['text_muted'])

        self.waveform_fig.tight_layout()
        self.waveform_canvas.draw()

    def play_audio(self):
        """Play audio"""
        if self.current_audio is None:
            messagebox.showwarning("No Audio", "Please load or record audio first!")
            return

        try:
            self.update_status("▶️ Playing audio...")
            sd.play(self.current_audio, self.current_sr)
            sd.wait()
            self.update_status("✓ Playback complete")
        except Exception as e:
            messagebox.showerror("Playback Error", f"Failed to play audio:\n{str(e)}")
            self.update_status("✗ Playback failed")

    def export_results(self):
        """Export results"""
        if not self.results_history:
            messagebox.showwarning("No Results", "No analysis results to export!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(self.results_history, f, indent=4)
                else:
                    with open(file_path, 'w') as f:
                        for i, result in enumerate(self.results_history, 1):
                            f.write(f"Analysis #{i}\n")
                            f.write("=" * 60 + "\n")
                            f.write(f"Timestamp: {result['timestamp']}\n")
                            f.write(f"Predicted Disease: {result['predicted_disease']}\n")
                            f.write(f"Confidence: {result['confidence']:.2f}%\n\n")
                            f.write("All Probabilities:\n")
                            for disease, prob in result['all_probabilities'].items():
                                f.write(f"  {disease}: {prob:.2f}%\n")
                            f.write("\n" + "=" * 60 + "\n\n")

                messagebox.showinfo("Success", f"Results exported to:\n{file_path}")
                self.update_status("✓ Results exported")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")

    def clear_all(self):
        """Clear all data"""
        if messagebox.askyesno("Clear All", "Are you sure you want to clear all data?"):
            self.current_audio = None
            self.current_sr = None
            self.results_history = []

            for widget in self.results_frame.winfo_children():
                widget.destroy()

            try:
                self.fig.clear()
                self.canvas.draw()
                self.waveform_fig.clear()
                self.waveform_canvas.draw()
            except Exception:
                pass

            self.analyze_btn.set_state(tk.DISABLED)
            self.play_btn.set_state(tk.DISABLED)
            self.export_btn.set_state(tk.DISABLED)

            self.update_status("✓ All data cleared")

    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=f"● {message}")
        self.root.update_idletasks()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    root = tk.Tk()
    app = RespiratoryDiseaseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()