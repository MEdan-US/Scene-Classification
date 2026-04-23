# Scene Classification Project

This project aims to perform scene classification using deep learning models. The system is designed to analyze scene patterns and classify them into different categories for accurate scene recognition.

## Dataset
[Scene Image Recognition-Kaggle](https://www.kaggle.com/datasets/jehanbhathena/scene-dataset)
## Features
- **Scene Data Processing**: Tailored for processing and analyzing scene data.
- **Deep Learning Models**: Utilizes ResNet-121 architecture.
- **Training on Labeled Data**: Built on a dataset labeled with scene categories, enabling the model to learn and predict scene conditions accurately.
- **Performance Metrics**: Model performance evaluated using accuracy, precision, recall, and F1-score.

## Getting Started
### Prerequisites
- Python 3.x
- Required Python packages: PyTorch, Streamlit, Scikit-learn, Pandas, NumPy

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/MEdan-US/scene_classification.git
   cd scene_classification
   ```
2. Create the virtual environment
   ```bash
   conda create --name scene_env -y
   conda activate scene_env
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To use the scene classification system:

1. Run the main script:
```
python -m streamlit run app.py
```
2. Follow the prompts to input scene data for classification.

## Acknowledgments
- Thanks to the original authors of machine learning and deep learning models for their significant contributions to classification tasks.
- We appreciate the open-source community for providing various libraries and frameworks that made this project possible.
