from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store the trained models and vectorizer
models = {
    'naive_bayes': MultinomialNB(),
    'svm': LinearSVC(random_state=42),
    'random_forest': RandomForestClassifier(random_state=42),
    'logistic_regression': LogisticRegression(random_state=42)
}
vectorizer = None
is_trained = False

def process_uploaded_file(file):
    """Process the uploaded CSV file and return preprocessed data"""
    # Save the file temporarily
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.csv')
    file.save(filepath)
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        data = None
        
        for encoding in encodings:
            try:
                data = pd.read_csv(filepath, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if data is None:
            raise ValueError("Unable to read the CSV file with supported encodings")

        # Validate the data format
        if len(data.columns) < 2:
            raise ValueError("CSV must have at least 2 columns: text and label")

        # Get column names from the first two columns
        text_col = data.columns[1]  # Assuming text is in the second column
        label_col = data.columns[0]  # Assuming label is in the first column

        # Create a new dataframe with renamed columns
        processed_data = pd.DataFrame({
            'text': data[text_col],
            'label': data[label_col]
        })

        # Convert labels to binary (0 or 1)
        unique_labels = processed_data['label'].unique()
        if len(unique_labels) != 2:
            raise ValueError("The label column must have exactly 2 unique values")

        # Map the first unique value to 0 and the second to 1
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        processed_data['label'] = processed_data['label'].map(label_map)

        return processed_data
    
    finally:
        # Clean up the temporary file
        if os.path.exists(filepath):
            os.remove(filepath)

def create_performance_plot(results):
    # Create a new figure for each plot to avoid threading issues
    plt.clf()
    fig = plt.figure(figsize=(10, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (model_name, scores) in enumerate(results.items()):
        plt.bar(x + i*width, 
               [scores['accuracy'], scores['precision'], 
                scores['recall'], scores['f1']], 
               width, 
               label=model_name)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width*1.5, metrics)
    plt.legend()
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()  # Close the buffer
    return plot_url

@app.route('/')
def home():
    return render_template('index.html', is_trained=is_trained)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400
    
    try:
        data = process_uploaded_file(file)
        return jsonify({'message': 'File uploaded successfully', 
                       'samples': len(data),
                       'spam_ratio': f"{(data['label'].mean() * 100):.2f}%"})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/train', methods=['POST'])
def train():
    global vectorizer, is_trained
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    try:
        # Process the uploaded file
        data = process_uploaded_file(file)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            data['text'], data['label'], test_size=0.2, random_state=42
        )
        
        # Initialize and fit the vectorizer
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train models and get performance metrics
        results = {}
        for model_name, model in models.items():
            # Train model
            model.fit(X_train_vec, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_vec)
            
            # Calculate metrics
            results[model_name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
        
        # Create performance plot
        plot_url = create_performance_plot(results)
        is_trained = True
        
        return jsonify({
            'results': results,
            'plot': plot_url,
            'is_trained': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    if not is_trained:
        return jsonify({'error': 'Please train the models first'}), 400
    
    text = request.json['text']
    model_name = request.json['model']
    
    try:
        # Vectorize the input text
        text_vec = vectorizer.transform([text])
        
        # Make prediction
        prediction = models[model_name].predict(text_vec)[0]
        
        return jsonify({
            'prediction': 'spam' if prediction == 1 else 'ham',
            'model_used': model_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)