# LungDL – Lung X-Ray Grad-CAM App

A FastAPI-based web service for analyzing lung X-ray images using Grad-CAM visualization. Built with Python, PyTorch, and FastAPI.

## Features

* Predict lung conditions from X-ray images.
* Visualize model predictions using Grad-CAM.
* Lightweight FastAPI server for local development.

---

## Prerequisites

* **OS**: Linux (tested on Arch Linux)
* **Python**: 3.13+
* **Node.js & npm**: for installing frontend dependencies
* **Git**: to clone the repository
* **uv CLI**: required for syncing Python environment. If not installed, follow the instructions at [https://uvcli.dev](https://uvcli.dev) to install it.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/blacklytning/lungDL
cd lungDL
```

### 2. Install Frontend Dependencies and Start Frontend

```bash
cd api
npm install
npm run dev
```

### 3. Sync Python Environment with `uv` CLI

```bash
uv sync
```

This will:

* Detect your Python interpreter (`CPython 3.13.7`)
* Create a virtual environment at `.venv`
* Install all required Python packages

### 4. Activate the Virtual Environment

For **Fish shell**:

```fish
source .venv/bin/activate.fish
```

For **Zsh or Bash**:

```zsh
source .venv/bin/activate
```

After activation, your prompt should show `(api)` or similar.

---

### 5. Start the FastAPI Development Server

```bash
fastapi dev
```

