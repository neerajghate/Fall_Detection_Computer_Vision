# Fall Detection System Using Computer Vision

## Overview

The elderly population living alone is increasing, making them particularly vulnerable to falls and emergencies. Traditional fall detection methods, such as wearable and ambient sensors, often face issues related to invasiveness, inaccessibility, and high cost. To address these concerns, this project presents a computer vision-based fall detection system that leverages Motion History Images (MHI) and a Raspberry Pi for accurate and non-intrusive fall detection.

## Key Features

- **Non-Intrusive Detection:** Uses computer vision to detect falls, avoiding the need for wearable or intrusive sensors.
- **High Accuracy:** Achieves fall detection accuracy between 98.5% and 99% using motion history images.
- **Real-Time Reporting:** The system provides immediate alerts to facilitate prompt medical intervention.
- **Privacy and Portability:** Designed with privacy in mind and can be used without relying on extensive storage or database systems.

## Technical Details

### Convolutional Neural Network (CNN)

The system employs a Convolutional Neural Network (CNN) to process video inputs and detect falls. CNNs are inspired by the visual cortex of the human brain and are effective at differentiating various features in images.

### Raspberry Pi

The Raspberry Pi serves as the core computing platform for the system. It is a compact, low-cost computer capable of running the fall detection software and handling video input from a connected camera.

### Motion History Image (MHI)

MHI is used to convert standard colored video frames into greyscale, which simplifies image processing and enhances detection accuracy. MHI allows for adjustable frame rates to balance processing speed and accuracy.

## Results

The fall detection model has demonstrated impressive performance:

- **Epoch 1/15:** Accuracy: 91.09%, Validation Accuracy: 98.54%
- **Epoch 2/15:** Accuracy: 97.76%, Validation Accuracy: 97.29%
- **Epoch 3/15:** Accuracy: 98.61%, Validation Accuracy: 96.46%
- **Epoch 10/15:** Accuracy: 99.64%, Validation Accuracy: 99.79%
- **Epoch 15/15:** Accuracy: 99.66%, Validation Accuracy: 100%

## Conclusion

The fall detection system effectively classifies human falls with approximately 99% accuracy, making it a reliable tool for real-time fall detection and emergency alerting.

## Getting Started

To use this system, follow the setup instructions provided in the repository and ensure you have a compatible camera and Raspberry Pi. For detailed setup and usage instructions, please refer to the [Installation Guide](#) in this repository.

## License

This project is licensed under the MIT License. See the [LICENSE](#) file for details.

