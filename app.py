import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\user\Downloads\DataSets-main (5)\DataSets-main\Iris.csv')

# Print the first few rows to verify the data was loaded correctly
print(df.head())
# Accessing specific columns
sepal_length = df['SepalLengthCm']
sepal_width = df['SepalWidthCm']
petal_length = df['PetalLengthCm']
petal_width = df['PetalWidthCm']
species = df['Species']

# Basic statistics
print(df.describe())
import matplotlib.pyplot as plt

# Scatter plot of sepal length vs. sepal width
plt.scatter(df['SepalLengthCm'], df['SepalWidthCm'])
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.show()

# Histogram of petal lengths
plt.hist(df['PetalLengthCm'])
plt.xlabel('PetalLengthCm')
plt.ylabel('Frequency')
plt.show()
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\user\Downloads\DataSets-main (5)\DataSets-main\Iris.csv')

# Print the column names
print(df.columns)
sepal_length = df['SepalLengthCm']


# Grouping by species and computing statistics
species_grouped = df.groupby('Species')
print(species_grouped.mean())


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the Iris dataset
    data = pd.read_csv(r"C:\Users\user\Downloads\DataSets-main (5)\DataSets-main\Iris.csv")

    # Prepare the dataset
    X = data.drop('Species', axis=1)
    y = data['Species']

    # Train a machine learning model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Get user input
    sepal_length = float(request.form['SepalLengthCm'])
    sepal_width = float(request.form['SepalWidthCm '])
    petal_length = float(request.form['PetalLengthCm'])
    petal_width = float(request.form['PetalWidthCm'])

    # Make a prediction
    input_data = [['SepalLengthCm, SepalWidthCm , PetalLengthCm,PetalWidthCm']]
    prediction = model.predict(input_data)[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
