# CSCI435 Food Waste AI Project – Development Setup Guide

## 1. Clone the Repository

Open a terminal and run:

```bash
git clone <repository-url>
cd <repository-folder>
```

Replace `<repository-url>` with the GitHub repository link.

---

## 2. Create a Python Virtual Environment

Navigate to the backend folder:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

---

## 3. Activate the Virtual Environment

### Windows (PowerShell)

```powershell
venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)

```cmd
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

---

## 4. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not been created yet, install:

```bash
pip install fastapi uvicorn sqlalchemy python-multipart pillow opencv-python ultralytics
```

Then generate the requirements file:

```bash
pip freeze > requirements.txt
```

---

## 5. Verify the Installation

Check that the following commands work:

```bash
python --version
pip --version
git --version
```

---

## 6. Run the Backend

From the `backend` folder:

```bash
uvicorn app.main:app --reload
```

If successful, open:

```
http://127.0.0.1:8000
```

Swagger API Documentation:

```
http://127.0.0.1:8000/docs
```

---

## 7. Git Workflow

Before starting work:

```bash
git pull
```

After completing your work:

```bash
git add .
git commit -m "Describe your changes"
git push
```

---

## 8. Branching Rules

* Never push directly to `main`.
* Create a feature branch for your task.
* Merge into `dev` through a Pull Request.
* Only merge into `main` after testing.

Example:

```bash
git checkout -b feature/backend-api
```

---

## 9. Project Folder Structure

```
backend/
│
├── app/
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── crud.py
│   ├── routers/
│   ├── services/
│   └── uploads/
│
├── requirements.txt
└── .env
```

---

## 10. Before You Start Coding

Make sure you can:

* Clone the repository
* Activate the virtual environment
* Install all dependencies
* Run the backend successfully
* Access `http://127.0.0.1:8000/docs`

If all of the above works, your development environment is ready.
