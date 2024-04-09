from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file
import mysql.connector, os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np

def load_and_preprocess_image(image_path):
    # Load the image file, resizing it to 224x224 pixels (as expected by MobileNet)
    img = load_img(image_path, target_size=(224, 224))
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    # Expand dimensions to match the shape expected by the pre-trained model: (1, 224, 224, 3)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    # Preprocess the image for the pre-trained model
    return preprocess_input(img_array_expanded)


def classify_leaf_image(image_path):
    # Load models and scaler (assumes these are loaded outside the function for efficiency)
    feature_extraction_model_path = 'feature_extractor.h5'
    feature_extraction_model = load_model(feature_extraction_model_path)

    scaler_path = 'scaler.save'
    scaler = joblib.load(scaler_path)

    svm_classifier_path = 'svm_model.pkl'
    svm_classifier = joblib.load(svm_classifier_path)

    # Process the image
    preprocessed_image = load_and_preprocess_image(image_path)

    # Extract features
    features = feature_extraction_model.predict(preprocessed_image)

    # Scale features
    scaled_features = scaler.transform(features.reshape(1, -1))

    # Predict with the SVM model
    predicted_class = svm_classifier.predict(scaled_features)
    
    # Map the predicted class to a label
    if predicted_class[0] == 0:
        result = "Healthy Leaf"
    elif predicted_class[0] == 1:
        result = "Spot Leaf"
    else:
        result = "Blotch Leaf"
    
    return result


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3307",
    database='turmeric_leaf'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password) VALUES (%s, %s)"
                values = (email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered! Please go to login section")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('home.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join('static/img/', fn)
        myfile.save(mypath)
        result = classify_leaf_image(mypath)
       
        return render_template('upload.html', path = mypath,result=result)
    return render_template('upload.html')



if __name__ == '__main__':
    app.run(debug = True)