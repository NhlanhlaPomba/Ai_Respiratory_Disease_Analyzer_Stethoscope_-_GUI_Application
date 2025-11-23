# AI Respiratory Disease Analyzer Stethoscope - GUI Application

![Project Banner](https://img.shields.io/github/languages/top/NhlanhlaPomba/Ai_Respiratory_Disease_Analyzer_Stethoscope_-_GUI_Application)
![License](https://img.shields.io/github/license/NhlanhlaPomba/Ai_Respiratory_Disease_Analyzer_Stethoscope_-_GUI_Application)
![Issues](https://img.shields.io/github/issues/NhlanhlaPomba/Ai_Respiratory_Disease_Analyzer_Stethoscope_-_GUI_Application)

## Overview

The **AI Respiratory Disease Analyzer Stethoscope - GUI Application** uses artificial intelligence to analyze recorded respiratory sounds via an easy-to-use graphical interface. The app helps clinicians, researchers, and students classify audio data for signs of respiratory disease, including wheezes, crackles, and abnormal breath sounds.

---

## Dataset Used

This application and its core model were trained using the [ICBHI Respiratory Sound Database](https://bhichallenge.med.auth.gr/), which contains annotated recordings of normal and abnormal respiratory sounds from clinical environments.

- **Dataset link:** [https://bhichallenge.med.auth.gr/](https://bhichallenge.med.auth.gr/)
- **Contents:** Over 5.5 hours of audio from 126 patients, annotated with disease markers (wheezes, crackles, normal, etc.).

---

## Features

- **AI-powered Disease Prediction:** Detects common respiratory anomalies.
- **Modern GUI:** Simple workflow for clinicians, researchers, and enthusiasts.
- **Multiple File Support:** Load or record `.wav`, `.mp3`, etc.
- **Result Exporting:** Save analysis and diagnostic results.
- **Educational Resources:** Learn details about lung sounds and disorders directly in-app.
- **.wav Audio Analysis:** Analyze respiratory sound recordings directly; no hardware is required.

---

## File Usage - Overview

Below is an explanation of the major Python and Arduino (`.ino`) files included in this repository, along with their typical use:

### Python Files

- **main.py:**  
  The main entry point and launcher for the GUI application.

- **respiratory_app.py:**  
  Another possible entry point for the application (use whichever is present in your repo). Handles initializing and showing the graphical user interface.

- **gui/\***  
  Contains code for the Graphical User Interface (GUI), such as window layouts, buttons, and menus.

- **audio_processing.py:**  
  Responsible for processing audio data, including extracting features (MFCCs, spectrograms), trimming, or cleaning audio files.

- **predict.py/model_inference.py:**  
  Implements AI model loading and prediction routines. Accepts processed audio and returns model predictions (disease labels).

- **train_model.py:**  
  Used to retrain or fine-tune the AI models on new or custom audio datasets.

- **utils.py:**  
  Provides utility/helper functions used throughout the project (data loading, saving, conversions, etc.).

- **export.py:**  
  Handles exporting of results, such as saving diagnostic reports as CSV or PDF.

### Arduino File

- **stethoscope_recorder.ino:**  
  (If present.) Code for an Arduino microcontroller to record respiratory sounds. It typically reads analog signals from an electronic stethoscope and streams or stores `.wav` data, communicating with the PC via USB or serial interface.

---

## Hardware Components (Optional)

You can run the software and analyze `.wav` files without any hardware integration. For real-time recording, build an electronic stethoscope using the following recommended components:

- **Microcontroller:** Arduino Uno/Nano, ESP32, or STM32
- **Microphone Module:** High-quality electret or MEMS microphone
- **Audio Amplifier:** To boost respiratory sound signal
- **SD Card Module:** For onboard storage of recordings (optional)
- **USB Data Interface or BLE:** To stream audio to your PC or mobile device
- **Miscellaneous:** Breadboard, wires, casing

> If you only want to analyze saved `.wav` recordings, hardware is NOT required.

---

## Step-by-Step Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/NhlanhlaPomba/Ai_Respiratory_Disease_Analyzer_Stethoscope_-_GUI_Application.git
cd Ai_Respiratory_Disease_Analyzer_Stethoscope_-_GUI_Application
```

### 2. Set Up Your Python Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Training Dataset (for retraining or testing)

1. Visit [ICBHI Respiratory Sound Database Download Page](https://bhichallenge.med.auth.gr/participate)
2. Create an account and download the dataset.
3. Unzip files into the `data/` directory.

### 5. Run the Application

```bash
python main.py
```
or
```bash
python respiratory_app.py
```
(Use whichever entry point matches your repository.)

---

## How to Use the App

1. **Record or Load Audio**  
   - Record via a stethoscope (hardware, if connected) or load an existing `.wav` file.
2. **Preprocess (Optional)**  
   - Apply filters or trim.
3. **Analyze Respiratory Sound**  
   - Click "Analyze" to run the AI model.
4. **View Results**  
   - View predicted classification and visual analysis.
5. **Save & Export**  
   - Save the report.

### Example Workflow

1. Launch the app.
2. Click "Load Audio" and select a `.wav` file.
3. Press "Analyze."
4. View results and export.

---

## Retraining Your Model (Advanced)

1. Download and place audio files in `data/`.
2. Run the training script:
    ```bash
    python train_model.py --data-dir ./data/ICBHI
    ```
3. Trained models are saved to `models/`.

---

## Project Structure

```
├── main.py                      # App launcher
├── respiratory_app.py           # Alternate launcher
├── gui/                         # GUI code
├── audio_processing.py          # Audio preprocessing
├── predict.py                   # Prediction routines
├── train_model.py               # Model training
├── utils.py                     # Utilities
├── export.py                    # Export features
├── stethoscope_recorder.ino     # (Arduino record controller—optional)
├── models/                      # Pre-trained AI models
├── data/                        # Respiratory datasets
├── requirements.txt             # Dependencies
├── README.md
└── LICENSE
```

---

## Contributing

1. Fork the repo, clone your fork.
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit changes (`git commit -am 'Add some fooBar'`)
4. Push (`git push origin feature/fooBar`)
5. Open a pull request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more info.

---

## Support & Issues

Open new issues [here](https://github.com/NhlanhlaPomba/Ai_Respiratory_Disease_Analyzer_Stethoscope_-_GUI_Application/issues) for bugs, requests, or help.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE).

---

## Acknowledgments

- [ICBHI Respiratory Sound Database](https://bhichallenge.med.auth.gr/)
- Open-source libraries: librosa, TensorFlow/PyTorch, PyQt5
- Inspiration from AI in healthcare and clinical research

---

*Empowering healthcare with AI-assisted respiratory diagnosis.*
