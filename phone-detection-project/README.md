# Phone Detection Project

This project is designed to recognize phones in a camera feed and calculate their size. It utilizes computer vision techniques and a pre-trained model to achieve accurate detection.

## Project Structure

```
phone-detection-project
├── src
│   ├── main.py               # Entry point of the application
│   ├── utils
│   │   └── image_processing.py # Utility functions for image processing
│   ├── models
│   │   └── phone_detector.py   # Phone detection model
│   └── config
│       └── settings.py        # Configuration settings
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore file
```

## Setup Instructions

1. Clone the repository:

   ```
   git clone <repository-url>
   cd phone-detection-project
   ```

2. Install the required dependencies:

   ```
   pip install --only-binary=opencv-python,ultralytics,opencv-python-headless -r requirements.txt
   ```

3. Configure the settings in `src/config/settings.py` as needed, including model paths and camera settings.

## Usage

To run the application, execute the following command:

```
python src/main.py
```

The application will initialize the camera feed, detect phones in the feed, and calculate their sizes.

## Functionality

- **Phone Detection**: The application uses a pre-trained model to identify phones in real-time from the camera feed.
- **Size Calculation**: Once a phone is detected, the application calculates its dimensions for further analysis.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
