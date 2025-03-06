from app import db
from flask_login import UserMixin
import datetime

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    stock_quantity = db.Column(db.Integer, default=0)
    price = db.Column(db.Float, nullable=False)
    image_url = db.Column(db.String(300), nullable=False)  
    barcode_image_url = db.Column(db.String(300), nullable=True)  # ✅ Store barcode image
    barcode = db.Column(db.String(100), unique=True, nullable=True)  # ✅ Store barcode

# ✅ Define the Detection Model
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    object_name = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# ✅ Define Database Model for Registered Faces
class RegisteredFace(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)



