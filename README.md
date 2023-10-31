# AI_Phase wise project submission
# Fake News Detection using NLP

This repository contains code for a Fake News Detection system using Natural Language Processing (NLP). The system uses a machine learning model trained on a labeled dataset to classify news articles as either fake or real.

Data Source: (https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

# Dependencies
Before running the code, make sure you have the following dependencies installed:
Python 3.6+

# Libraries:
pandas
scikit-learn
nltk
numpy
matplotlib
# You can install these libraries using the following command:
# pip install pandas scikit-learn nltk numpy matplotlib

# Running the Code
Clone the repository to your local machine:
git clone :(https://github.com/Sabithrija/Phase2.git)
# Navigate to the project directory:

cd fake_news_detection
Place your dataset(Kaggle.com)in the data/ directory.

# Run the preprocess.py script to clean and preprocess the data:
python preprocess.py

# Train the model using the train_model.py script:
python train_model.py

Once the model is trained, you can use it for predictions. You can run the predict.py script with your own text input, or use the model in your own applications.
python predict.py --text "Your news article text goes here"
# Additional Notes

The model used in this code is a simple example. Depending on your specific needs, you may want to experiment with different algorithms, feature extraction techniques, or fine-tune hyperparameters.

Make sure to cite the source of the dataset you use (if it's not a synthetic dataset).

Feel free to customize and extend the code to suit your specific requirements.

## Dataset

The dataset used for this Fake News Detection project https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset provided by Kaggle. This dataset contains a collection of labeled news articles, where each article is categorized as either "fake" or "real".

### Description

The dataset consists of 3 news articles. Each sample is accompanied by its corresponding label, making it suitable for supervised learning tasks like classification.

### Source Information

- **Name**: [Source name:Kaggle.com]( https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset-)
- **Link**: [ https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset]
