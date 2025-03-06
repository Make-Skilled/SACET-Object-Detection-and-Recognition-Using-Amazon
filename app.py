from flask import Flask
from flask_sqlalchemy import SQLAlchemy, pagination
from flask_login import LoginManager
from flask_mail import Mail, Message

import os

# Initialize Flask App
app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")

UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Configure Flask-Mail (Email Alerts)
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "makeskilledtest@gmail.com"
app.config["MAIL_PASSWORD"] = "mqqk slyn vvix uxwu"
app.config["MAIL_DEFAULT_SENDER"] = "makeskilledtest@gmail.com"

mail = Mail(app)

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Database
db = SQLAlchemy(app)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
Admin='maddy@makeskilled.com'

# ✅ Function to Send Email Alert
def send_alert_email(unknown_face_path):
    try:
        print("📤 Sending Email...")
        msg = Message("⚠️ Unrecognized Person Detected", recipients=[Admin])
        msg.body = "An unrecognized person has been detected! See attached face image."

        with open(unknown_face_path, "rb") as f:
            msg.attach("unknown_face.jpg", "image/jpeg", f.read())

        mail.send(msg)
        print("✅ Email Sent Successfully!")

    except Exception as e:
        print(f"❌ Email Sending Failed: {e}")


# Import routes AFTER initializing app to avoid circular imports
from routes import *

# ✅ FIX: Use the application context before calling `db.create_all()`
if __name__ == "__main__":
    with app.app_context():  # ✅ Ensure the app context is set up
        db.create_all()
    app.run(debug=True,port=9001)
