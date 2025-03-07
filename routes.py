from flask import render_template, redirect, url_for, flash, request,send_from_directory,session,jsonify,Response
from werkzeug.utils import secure_filename
from flask_login import login_user, logout_user, login_required
from app import app, db, login_manager, send_alert_email
from dbmodels import User, Product, Detection, RegisteredFace
import boto3
import cv2
import torch
import time
import numpy as np
import sys
sys.path.append('./yolov5')  # Add YOLOv5 directory to Python's path
import os
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression
import base64
import random
import datetime

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# AWS Rekognition Client
rekognition_client = boto3.client("rekognition", region_name="us-east-1")  # Change region if needed

# YOLOv5 Configuration
device = select_device("")
yolo_model = DetectMultiBackend("yolov5s.pt", device=device)  # Load YOLOv5 model

# Ensure detected_faces directory exists
DETECTED_FACES_DIR = "static/detected_faces"
if not os.path.exists(DETECTED_FACES_DIR):
    os.makedirs(DETECTED_FACES_DIR)

COLORS = {}
def get_color(cls_id):
    if cls_id not in COLORS:
        COLORS[cls_id] = [random.randint(0, 255) for _ in range(3)]  # Generate random RGB color
    return COLORS[cls_id]

 # ✅ Define Cooling Period (Seconds)
COOLING_PERIOD = 30  # Change this to modify the time interval

os.environ["DYLD_LIBRARY_PATH"] = os.popen("brew --prefix zbar").read().strip() + "/lib"

from pyzbar.pyzbar import decode

def detect_barcode(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return None

    barcodes = decode(img)

    if not barcodes:
        return None  # No barcode found

    barcode_data = [barcode.data.decode("utf-8") for barcode in barcodes]
    return barcode_data  # Returns a list of detected barcodes

# ✅ Function to Detect Objects in an Image
def detect_objects_in_image(image_path):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()  # ✅ Ensure image is read as bytes

    if not image_bytes:
        print("Error: Image file is empty.")
        return []

    try:
        response = rekognition_client.detect_labels(
            Image={"Bytes": image_bytes},  # ✅ Pass image bytes
            MaxLabels=10,
            MinConfidence=70
        )
        detected_objects = [label["Name"] for label in response["Labels"]]
        return detected_objects

    except Exception as e:
        print("Error in AWS Rekognition:", str(e))
        return []


def compare_faces(registered_faces, detected_face_path):
    print('Calling AWS Compare Faces API')
    with open(detected_face_path, "rb") as detected_face:
        detected_bytes = detected_face.read()

    for face in registered_faces:
        with open(face.image_path, "rb") as registered:
            registered_bytes = registered.read()

        response = rekognition_client.compare_faces(
            SourceImage={"Bytes": registered_bytes},
            TargetImage={"Bytes": detected_bytes},
            SimilarityThreshold=85
        )

        if response["FaceMatches"]:
            print("Matched")
            return True  # ✅ Match found
        
    print("No Match")
    return False  # ❌ No match found

# ✅ Function to Find a Similar Product Based on Objects
def find_similar_product(image_path):
    all_products = Product.query.all()

    for product in all_products:
        existing_image_path = os.path.join("static", product.image_url)
        if not os.path.exists(existing_image_path):
            continue  # Skip if the file doesn't exist

        # ✅ Compare Image Features using ORB
        similarity_score = match_images(image_path, existing_image_path)
        print(f"Comparing {product.name}: {similarity_score} matches found.")

        if similarity_score > 30:  # ✅ Set threshold for similarity
            return product  # ✅ Return matching product

    return None  # No match found

def match_images(image1_path, image2_path, threshold=30):
    """
    Compares two images using ORB feature matching and returns the number of matches.
    """
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error: One or both images could not be loaded.")
        return 0

    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    if descriptors1 is None or descriptors2 is None:
        return 0  # No descriptors found

    # Match features using Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (lower distance = better match)
    matches = sorted(matches, key=lambda x: x.distance)

    return len(matches)  # Return number of matched features

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# User Loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
@login_required
def dashboard():
    search_query = request.args.get("search", "").strip()
    category_filter = request.args.get("category", "")
    sort_by = request.args.get("sort", "newest")  # Default to newest
    page = request.args.get("page", 1, type=int)
    per_page = 5  # Items per page

    # Start with all products
    products = Product.query

    # Apply search filter (case-insensitive search by name)
    if search_query:
        products = products.filter(Product.name.ilike(f"%{search_query}%"))

    # Apply category filter
    if category_filter:
        products = products.filter(Product.category == category_filter)

    # Apply sorting
    if sort_by == "newest":
        products = products.order_by(Product.id.desc())  # Newest first
    elif sort_by == "oldest":
        products = products.order_by(Product.id.asc())  # Oldest first
    elif sort_by == "price_asc":
        products = products.order_by(Product.price.asc())  # Price low to high
    elif sort_by == "price_desc":
        products = products.order_by(Product.price.desc())  # Price high to low
    elif sort_by == "stock_asc":
        products = products.order_by(Product.stock_quantity.asc())  # Stock low to high
    elif sort_by == "stock_desc":
        products = products.order_by(Product.stock_quantity.desc())  # Stock high to low

    # Apply pagination
    products = products.paginate(page=page, per_page=per_page, error_out=False)

    return render_template("dashboard.html", products=products)

@app.route("/live_feed")
@login_required
def live_feed():
    """ Start live camera feed, detect objects with YOLOv5, and update stock in database """
    cap = cv2.VideoCapture(0)

    frame_skip = 30  # Process 1 frame per second (~30 FPS)
    last_detected_objects = {}  # Caching detected objects
    detection_interval = 3  # Call AWS Rekognition every 3 seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame for YOLOv5
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # Normalize

        # YOLOv5 Object Detection
        with torch.no_grad():
            pred = yolo_model(img.unsqueeze(0))

        # Apply Non-Maximum Suppression (NMS)
        pred = non_max_suppression(pred, 0.5, 0.45)[0]

        # Extract detected objects
        detected_objects = []
        if pred is not None and len(pred):
            for det in pred:
                _, _, _, _, conf, cls = det.tolist()
                label = yolo_model.names[int(cls)]
                detected_objects.append(label)

        print("YOLO Detected Objects:", detected_objects)

        # If detected objects are different, call AWS Rekognition
        if detected_objects and detected_objects != last_detected_objects and (time.time() - last_detected_objects.get("timestamp", 0)) > detection_interval:
            last_detected_objects = {"objects": detected_objects, "timestamp": time.time()}  # Update cache

            # Convert frame to bytes for AWS Rekognition
            _, buffer = cv2.imencode(".jpg", frame)
            image_bytes = buffer.tobytes()

            # Call AWS Rekognition
            response = rekognition_client.detect_labels(
                Image={"Bytes": image_bytes}, MaxLabels=10, MinConfidence=80
            )

            rekognition_detected = [label["Name"] for label in response["Labels"]]
            print("AWS Rekognition Detected:", rekognition_detected)

            # Update stock in the database
            for obj in rekognition_detected:
                product = Product.query.filter_by(name=obj).first()
                if product:
                    product.stock_quantity += 1
                else:
                    new_product = Product(name=obj, stock_quantity=1, detected_objects=obj)
                    db.session.add(new_product)
                db.session.commit()

        # Display the live feed
        cv2.imshow("Live Feed - YOLOv5 + AWS Rekognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for("dashboard"))

# 📌 LOGOUT ROUTE
@app.route("/logout")
@login_required
def logout():
    logout_user()
    if 'scanned_barcode' in session:
        session.pop("scanned_barcode", None)
    flash("Logged out successfully!", "success")
    return redirect(url_for("index"))

# 📌 LOGIN ROUTE
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email, password=password).first()

        if user:
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password. Try again.", "error")

    return render_template("login.html")

# 📌 SIGNUP ROUTE
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        # Check if email already exists
        if User.query.filter_by(email=email).first():
            flash("Email already exists! Try logging in.", "error")
            return redirect(url_for("signup"))

        # Create new user
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash("Signup successful! You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

# ✅ Add Product Route (Uploads Product & Barcode Image)
@app.route("/add_product", methods=["GET", "POST"])
@login_required
def add_product():
    if request.method == "POST":
        category = request.form["category"]
        name = request.form["name"]
        description = request.form["description"]
        quantity = int(request.form["quantity"])
        price = float(request.form["price"])
        barcode = session.get("scanned_barcode")  # ✅ Use scanned barcode
        image_data = request.form.get("imageData")  # ✅ Captured image from camera
        product_image = request.files.get("product_image")  # ✅ Uploaded image

        if not barcode:
            flash("Error: Barcode not found!", "error")
            return redirect(url_for("scan_barcode"))

        # ✅ Save product image (Either uploaded or captured)
        if image_data:
            image_data = image_data.replace("data:image/png;base64,", "")
            image_bytes = base64.b64decode(image_data)
            image_filename = f"product_{barcode}.png"
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
        elif product_image:
            image_filename = secure_filename(product_image.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            product_image.save(image_path)
        else:
            flash("Error: No image provided!", "error")
            return redirect(url_for("add_product"))

        # ✅ Store image path
        image_url = f"uploads/{image_filename}"

        # ✅ Save product
        new_product = Product(category=category, name=name, description=description, stock_quantity=quantity, price=price, image_url=image_url, barcode=barcode)
        db.session.add(new_product)
        db.session.commit()
        
        # ✅ Clear session barcode after product is added
        session.pop("scanned_barcode", None)

        flash(f"New product added successfully!", "success")
        return redirect(url_for("dashboard"))

    if request.method == "GET":
        barcode = session.get("scanned_barcode")
        if not barcode:
            flash("Error: Barcode not found!", "error")
            return redirect(url_for("scan_barcode"))
    
    return render_template("add_product.html")

# ✅ Edit Product Route (Include Price)
@app.route("/edit_product/<int:product_id>", methods=["GET", "POST"])
@login_required
def edit_product(product_id):
    product = Product.query.get_or_404(product_id)

    if request.method == "POST":
        product.category = request.form["category"]
        product.name = request.form["name"]
        product.description = request.form["description"]
        product.stock_quantity = int(request.form["quantity"])
        product.price = float(request.form["price"])

        image_data = request.form.get("imageData")  # ✅ Captured image from camera
        product_image = request.files.get("product_image")  # ✅ Uploaded image

        # ✅ Process New Product Image (Either from Camera Capture OR Upload)
        if image_data:
            image_data = image_data.replace("data:image/png;base64,", "")
            image_bytes = base64.b64decode(image_data)
            image_filename = f"product_{product.barcode}.png"
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            product.image_url = f"uploads/{image_filename}"
        elif product_image:
            image_filename = secure_filename(product_image.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            product_image.save(image_path)
            product.image_url = f"uploads/{image_filename}"
        else:
            image_filename = product.image_url  # Keep existing image if no new image is uploaded        

        # ✅ Save Changes to Database
        db.session.commit()
        flash("Product updated successfully!", "success")
        return redirect(url_for("dashboard"))

    return render_template("edit_product.html", product=product)

# ✅ Delete Product Route
@app.route("/delete_product/<int:product_id>", methods=["GET"])
@login_required
def delete_product(product_id):
    product = Product.query.get_or_404(product_id)
    db.session.delete(product)
    db.session.commit()
    flash("Product deleted successfully!", "success")
    return redirect(url_for("dashboard"))

# ✅ Route to Show Live Barcode Scanner Page
@app.route("/scan_barcode", methods=["GET"])
@login_required
def scan_barcode():
    return render_template("scan_barcode.html")

# ✅ Process Image and Detect Barcode
@app.route("/process_barcode_image", methods=["POST"])
@login_required
def process_barcode_image():
    image_data = request.form.get("imageData")

    if not image_data:
        flash("No image received!", "error")
        return redirect(url_for("scan_barcode"))

    try:
        # ✅ Decode Base64 Image
        image_data = image_data.replace("data:image/png;base64,", "")
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            flash("Error decoding image!", "error")
            return redirect(url_for("scan_barcode"))

        # ✅ Convert Image to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ✅ Detect Barcodes
        barcodes = decode(gray)
        print(barcodes)

        if not barcodes:
            flash("No barcode detected! Please try again.", "error")
            return redirect(url_for("scan_barcode"))

        barcode_text = barcodes[0].data.decode("utf-8")  # Get first barcode

        # ✅ Check if the product exists in the database
        product = Product.query.filter_by(barcode=barcode_text).first()
        if product:
            flash(f"Product found: {product.name}", "success")
            return redirect(url_for("edit_product", product_id=product.id))
        
        # ✅ Store barcode in session for later use
        session["scanned_barcode"] = barcode_text
        flash("New barcode detected! Please enter product details.", "info")
        return redirect(url_for("add_product"))

    except Exception as e:
        flash(f"Error processing barcode: {str(e)}", "error")
        return redirect(url_for("scan_barcode"))

# ✅ Route to Show Live Object Detection Page
@app.route("/live_detection")
def live_detection():
    return render_template("live_detection.html")


@app.route("/detect_objects", methods=["POST"])
def detect_objects():
    try:
        data = request.get_json()
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "No image received"}), 400

        # ✅ Convert Base64 Image to OpenCV format
        image_data = image_data.replace("data:image/jpeg;base64,", "")
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # ✅ YOLOv5 Object Detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize
        img_tensor = img_tensor.to(device)  # Move tensor to same device as model
        
        with torch.no_grad():
            pred = yolo_model(img_tensor, augment=False)
            pred = non_max_suppression(pred, conf_thres=0.5)[0]  # ✅ Apply Threshold (conf ≥ 0.5)

        # ✅ Get Current Time
        current_time = datetime.datetime.utcnow()
        detected_objects = []
        registered_faces = RegisteredFace.query.all()

        if pred is not None and len(pred) > 0:
            for det in pred:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                
                if conf < 0.5:  # ✅ Ignore low-confidence detections
                    continue

                cls_id = int(cls)
                object_name = yolo_model.names[cls_id]
                color = get_color(cls_id)  # Assign unique color per object type

                # ✅ Always Draw Bounding Box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(img, f"{object_name} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # ✅ Always Log Object Detection to Database
                db.session.add(Detection(object_name=object_name, confidence=conf))
                detected_objects.append(object_name)

                # ✅ Process Faces If Object is "Person" - Only during cooling period
                if object_name == "person":
                    face_img = img[int(y1):int(y2), int(x1):int(x2)]
                    face_path = os.path.join(DETECTED_FACES_DIR, f"{datetime.datetime.utcnow().timestamp()}.jpg")
                    cv2.imwrite(face_path, face_img)

                    # ✅ Check Cooling Period for Face Comparison
                    last_alert_time = db.session.query(db.func.max(Detection.timestamp)).filter_by(object_name="UNKNOWN_PERSON").scalar()
                    if not last_alert_time or (current_time - last_alert_time).total_seconds() >= COOLING_PERIOD:
                        if not compare_faces(registered_faces, face_path):
                            send_alert_email(face_path)  # 🚨 Send alert if face is unknown
                            db.session.add(Detection(object_name="UNKNOWN_PERSON", confidence=1.0))

        db.session.commit()

        # ✅ Always Return Processed Image with Bounding Boxes
        _, buffer = cv2.imencode(".jpg", img)
        return Response(buffer.tobytes(), mimetype="image/jpeg")

    except Exception as e:
        print(f"Error in detect_objects: {str(e)}")  # Add logging
        return jsonify({"error": str(e)}), 500
    
# ✅ Route to Display Stored Detections
@app.route("/detection_log")
def detection_log():
    detections = Detection.query.order_by(Detection.timestamp.desc()).all()
    return render_template("detection_log.html", detections=detections)

# ✅ Route for Face Registration
@app.route("/register_face", methods=["GET", "POST"])
def register_face():
    if request.method == "POST":
        name = request.form["name"]
        image = request.files.get("face_image")
        captured_image = request.form.get("captured_image")

        # ✅ Save Image from Camera Capture
        if captured_image:
            image_data = captured_image.replace("data:image/jpeg;base64,", "")
            image_bytes = base64.b64decode(image_data)
            image_path = f"static/registered_faces/{name}.jpg"

            with open(image_path, "wb") as f:
                f.write(image_bytes)

        # ✅ Save Image from File Upload
        elif image:
            image_path = f"static/registered_faces/{name}.jpg"
            image.save(image_path)

        else:
            return jsonify({"error": "No image provided"}), 400

        # ✅ Store in Database
        new_face = RegisteredFace(name=name, image_path=image_path)
        db.session.add(new_face)
        db.session.commit()
        flash("Face Registered Successfully!", "info")

    return render_template("register_face.html")

@app.route("/manage_faces")
def manage_faces():
    faces = RegisteredFace.query.all()
    return render_template("manage_faces.html", faces=faces)

@app.route("/delete_face/<int:face_id>", methods=["POST"])
def delete_face(face_id):
    face = RegisteredFace.query.get(face_id)
    if not face:
        flash("Face not found!", "danger")
        return redirect(url_for("manage_faces"))

    # ✅ Remove the face image file from storage
    if os.path.exists(face.image_path):
        os.remove(face.image_path)

    # ✅ Delete face entry from the database
    db.session.delete(face)
    db.session.commit()

    flash("Face deleted successfully!", "success")
    return redirect(url_for("manage_faces"))