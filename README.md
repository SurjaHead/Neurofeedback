<div align="center">
<h1 align="center">Neurofeedback EEG Stream</h1>

  <p align="center">
    Live EEG data visualization and concentration/relaxation level estimation using Brainflow.
  </p>
</div>

## Table of Contents

  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#key-features">Key Features</a></li>
      </ul>
    </li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>

## About The Project

This project provides a real-time visualization of EEG data streamed from a Brainflow-compatible EEG device. It processes the raw EEG signals to estimate the user's concentration and relaxation levels, providing feedback based on these metrics. The application utilizes signal processing techniques like bandpass and bandstop filters to clean the EEG data and extracts relevant features for machine learning models.

### Key Features

- **Live EEG Data Visualization:** Displays real-time EEG data from four channels in a plot that updates dynamically.
- **Signal Processing:** Applies bandpass and bandstop filters to remove noise and isolate specific frequency bands in the EEG signal.
- **Concentration and Relaxation Estimation:** Uses pre-trained machine learning models to estimate the user's concentration and relaxation levels based on EEG features.
- **Real-time Feedback:** Provides textual feedback to the user, encouraging them to concentrate more or relax more based on the estimated levels.

## Built With

- [Python](https://www.python.org/)
- [Brainflow](https://brainflow.readthedocs.io/en/stable/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)

## Getting Started

To get started with the project, you'll need to install the necessary dependencies and configure the Brainflow library to connect to your EEG device.

### Prerequisites

- Python 3.6 or higher
- Pip package installer
- A Brainflow-compatible EEG device (e.g., OpenBCI)
- Brainflow SDK installed

  ```sh
  pip install brainflow
  pip install pandas
  pip install matplotlib
  pip install numpy
  ```

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/surjahead/neurofeedback.git
   ```
2. Navigate to the project directory:
   ```sh
   cd neurofeedback
   ```
3. Install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```
   *Note: A requirements.txt file was not provided in the repository.  The packages listed in the prerequisites should be added to a requirements.txt file.*
4.  Modify the `main.py` file to match your board ID and serial port:
    ```python
    params = BrainFlowInputParams()
    params.board_id = 1  # Replace with your board ID
    board_id = 1 # Replace with your board ID
    params.serial_port = 'COM5'  # Replace with your serial port
    ```
5. Run the `main.py` script:
   ```sh
   python main.py
   ```

## Acknowledgments

- This project was created by using the OpenBCI Ganglion Board and the BrainFlow documentaiton.
