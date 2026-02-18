from tensorflow.keras.models import load_model
from flask import Flask, render_template, request,g, session, redirect, url_for, flash
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
import pyrebase
import os

#Configuration of firebase databse
Config = {
  "apiKey": "AIzaSyCLlpAOzJraCP7TLDMp_zkbaqsPOaawKQE",
  "authDomain": "diabetic-retinopathy-29d23.firebaseapp.com",
  "projectId": "diabetic-retinopathy-29d23",
  "storageBucket": "diabetic-retinopathy-29d23.firebasestorage.app",
  "messagingSenderId": "1027285319021",
  "appId": "1:1027285319021:web:25386667de7f57bfff2e64",
  "measurementId": "G-BT5MVBBTC9",
  "databaseURL":"https://diabetic-retinopathy-29d23-default-rtdb.firebaseio.com/"
};

#flask app decleration
app=Flask(__name__)
app.secret_key= "hiuffhhwiuf"

#Firebase database intilization
firebase_ = pyrebase.initialize_app(Config)
db = firebase_.database()

#load the trained model
model = load_model("Updated-Xception-diabetic-retinopathy.h5")

#upload folder for uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#render the home page 
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("home.html")

#registration route
@app.route("/register", methods=['GET', 'POST'])
def regis():
    if request.method == "POST":
        Name = request.form
        db.child("All Data").push(Name)
        return redirect("login")
    return render_template("register.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        session.pop("user", None)
        email = request.form.get("email")
        password = request.form.get("password")
        user_found = False
        data1 = db.child("All Data").get()
        for user in data1.each():
            user_data = user.val()
            if email == user_data["email"] and password == user_data["password"]:
                session['user'] = user_data['name']
                user_found = True
                #return render_template("prediction.html", name= user_data["name"])
                return redirect(url_for("pred"))
        if not user_found:
            flash(f"Invalid user_ID and password") 
            return render_template("login.html")      
    return render_template("login.html")

@app.route("/prediction", methods=['GET', 'POST'])
def pred():
    name = session['user']
    if request.method == "POST":

        file = request.files["file"]

        if file.filename == "":
            return "No file selected"

        # Save file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        img_path = filepath  # change this

        img = image.load_img(img_path, target_size=(299, 299))
        img_array = image.img_to_array(img)

        # Expand dimensions (model expects batch format)
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess for Xception
        img_array = preprocess_input(img_array)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        confidence = confidence * 100

        class_labels = {0: "No_DR", 1: "Mild_DR", 2: "Moderate_DR", 3: "Severe_DR", 4: "Proliferative_DR"}

        predicted_class = class_labels[predicted_class]
        return render_template("prediction.html", name = name, prediction = predicted_class, confidence=confidence, uploaded_image=filepath)

    return render_template("prediction.html", name = name)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

if __name__ == "__main__":
    app.run(debug=True)