# 📦 Warehouse Object Detection & Face Recognition System

This project is a **Flask-based AI-powered warehouse monitoring system** that integrates **YOLOv5 for object detection** and **AWS Rekognition for face recognition**. It includes **live camera streaming, barcode scanning, object tracking, and email alerts for unrecognized faces**.

## 🚀 Features
- ✅ **Live Object Detection** using **YOLOv5**
- ✅ **Face Recognition** with AWS Rekognition
- ✅ **Barcode Scanning** using `pyzbar`
- ✅ **Live Streaming via Browser Camera**
- ✅ **Detection Logs with Pagination & Search**
- ✅ **Automatic Email Alerts for Unrecognized Faces**
- ✅ **Manage Registered Faces (Add/Delete)**
- ✅ **Cooling Period Logic to Reduce API Calls**
- ✅ **Optimized UI with Tailwind CSS**

---

## 🏗️ **Installation**
Follow these steps to set up and run the project.

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-repo/warehouse-detection.git
cd warehouse-detection
```

### 2️⃣ **Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Set Up AWS Rekognition**
- Create an **AWS IAM user** with **Rekognition** and **SES (for emails)** permissions.
- Configure AWS credentials:
```bash
aws configure
```
- Enter your **AWS Access Key**, **Secret Key**, and **Region**.

### 5️⃣ **Run the Application**
```bash
flask run
```
The app will be accessible at:  
📌 **http://127.0.0.1:5000**

---

## 📷 **Live Camera Streaming**
- Go to **"Live Object Detection"** in the dashboard.
- The browser **camera stream will start** and detect objects in real-time.

## 🧑‍💻 **Face Recognition & Alerts**
- Register a face under **"Register Face"**.
- If an **unknown face appears**, an **email alert** will be sent.

## 📦 **Barcode Scanning**
- Use **"Scan Barcode"** to scan a product.
- If a barcode **already exists**, product info is retrieved.
- If it's **new**, the system prompts for product details.

---

## 🛠 **Tech Stack**
- **Backend**: Flask, Flask-Mail, Boto3 (AWS)
- **Frontend**: Tailwind CSS, JavaScript, HTML
- **AI/ML**: YOLOv5 (Object Detection), AWS Rekognition (Face Matching)
- **Database**: SQLite

---

## 🎯 **Upcoming Features**
- 📌 **Real-time notifications** via WebSockets
- 📌 **Admin Dashboard for Advanced Analytics**
- 📌 **Export Detection Logs to CSV/PDF**
- 📌 **Automatic Warehouse Stock Updates**

---

## 📩 **Contact**
🔹 **Author**: [Madhu Parvathaneni](https://linkedin.com/in/MadhuPIoT)  
🔹 **Email**: maddy@makeskilled.com  
🔹 **GitHub**: [your-github-link](https://github.com/maddydevgits)

🚀 **Happy Coding!** 🎯
