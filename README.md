# AI Clothing Classifier - Web Interface

This project extends an existing AI clothing classification system (which uses K-means clustering and KNN) with a modern web interface. The original system classifies clothing items by color and shape, and this web interface allows users to interact with the system through a friendly UI instead of a command-line interface.

## Features

- Modern, responsive web interface
- Filter clothing items by:
  - Type (shirts, flip-flops, etc.)
  - Colors (multiple selection)
  - Combined filters
- Switch between grid and list views
- Detailed item information display

## Architecture

- **Backend**: Flask API that connects to the existing AI classification system
- **Frontend**: HTML/CSS/JavaScript with a clean, modern interface
- **Models**:
  - K-means for color classification
  - KNN for shape classification

## Setup and Installation

1. Make sure you have Python 3.6+ installed
2. Clone this repository
3. Install dependencies:

```
pip install -r requirements.txt
```

4. Ensure the `/images` directory contains the training and test datasets

## Running the Application

1. Start the Flask server:

```
python app.py
```

2. Open a web browser and go to:

```
http://127.0.0.1:5000
```

## Project Structure

```
/
├── app.py                # Flask application
├── my_labeling.py        # Original classification system
├── Kmeans.py             # K-means implementation
├── KNN.py                # KNN implementation
├── utils_data.py         # Data handling utilities
├── templates/            # HTML templates
│   └── index.html        # Main page
├── static/               # Static assets
│   ├── css/              # CSS styles
│   │   └── style.css     # Main stylesheet
│   ├── js/               # JavaScript files
│   │   └── app.js        # Frontend functionality
│   └── temp_imgs/        # Temporary image storage
└── images/               # Dataset directory
    ├── train/            # Training images
    ├── test/             # Test images
    └── gt.json           # Ground truth data
```

## API Endpoints

- `GET /api/options` - Get available filter options (shapes and colors)
- `POST /api/filter/color` - Filter items by color
- `POST /api/filter/shape` - Filter items by shape
- `POST /api/filter/combined` - Filter items by both color and shape 