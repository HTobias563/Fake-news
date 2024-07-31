# Fake News Detection

This project aims to detect fake news articles using a DistilBERT-based model. The model is trained on the WELFake Dataset and can classify news articles as real or fake. Additionally, a Flask web application is provided to allow users to check any article from the internet.

## Dataset

- **WELFake Dataset:** [Download from Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data)

## Key Features

1. **DistilBERT Model:** Utilizes a pre-trained DistilBERT model fine-tuned for fake news detection.
2. **Data Preprocessing:** Includes scripts to preprocess the WELFake Dataset.
3. **Training and Evaluation:** Code to train the model and evaluate its performance on a test set.
4. **Flask Web App:** A web application to predict the authenticity of any news article by providing its URL.

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/HTobias563/Fake-news.git
    cd fake-news
    ```

2. **Install the Required Packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Flask Web Application

1. Ensure you have installed the required packages as described in the installation section.
2. Start the Flask app:
    ```bash
    python app.py
    ```
3. Open a web browser and go to `http://127.0.0.1:5000` to access the application.

### Testing the Model

1. **Prepare Test Data:** Ensure you have test data in the correct format.
2. **Run the Testing Script:**
    ```bash
    python testing.py
    ```
   This will evaluate the model on the test data and print performance metrics.

## Files

- `app.py`: Contains the Flask web application code.
- `DistilBertModel.py`: Contains the model definition and training code.
- `testing.py`: Script to evaluate the model on test data.
- `templates/index.html`: HTML template for the Flask web application.
- `requirements.txt`: List of required Python packages.

## Contributing

Feel free to open an issue or submit a pull request if you have suggestions or improvements.


