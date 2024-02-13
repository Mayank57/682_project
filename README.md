# Smart Surveillance: Person Recognition Using Textual Description

## Introduction
This project aims to contribute to surveillance technology by introducing an automated person recognition system. Leveraging deep learning, specifically Faster R-CNN and ResNet50, the system enhances the identification and localization of individuals in CCTV footage, addressing the inefficiencies of manual monitoring.

## Installation
```bash
git clone <repository-url>
cd Smart-Surveillance
cd "Attribute recognition"
pip install -r requirements.txt
```
Usage
To start the person recognition process, execute:
```bash
python person_recognition.py --input <path_to_video>
```


## Features
  - Automated Person Detection: Utilizes Faster R-CNN for real-time object detection in video footage.
  - Attribute Recognition: Employs fine-tuned ResNet50 on the PETA dataset for precise attribute matching, including clothing and accessories.
  - Query-Based Person Mapping: Processes natural language queries to match textual descriptions with visual attributes, enhancing the search and identification process.

## Dataset
The project uses two main datasets:

  - UCF Crime Dataset: For analyzing real-world surveillance videos.
  - PETA Dataset: Known for its detailed pedestrian attribute annotations, aiding in attribute recognition.

**Please go through the pdf document for further information regarding the project including the results and accuracies.**

## License
Distributed under the MIT License. See LICENSE for more information.

## Acknowledgements
This project builds upon existing research in the fields of computer vision and machine learning, aiming to provide a practical solution for law enforcement and public safety.

