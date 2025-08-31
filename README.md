# Qode-Advisors-LLP-Project
Qode Advisors LLP

Twitter/X Stock Market Data Scraper
A Python script to scrape Indian stock market tweets from Twitter/X using snscrape library. This tool extracts tweets containing popular stock market hashtags and saves them to a CSV file for analysis.

Prerequisites
This project requires Python 3.11.9 specifically due to snscrape compatibility requirements.

Installation:

    Step 1: Install Python 3.11.9
        Visit: https://www.python.org/downloads/release/python-3119/
        Scroll down to the "Files" section
        Download the appropriate installer:
            For 64-bit Windows: python-3.11.9-amd64.exe
            For 32-bit Windows: python-3.11.9.exe

        Run the installer and follow the installation wizard

    Step 2: Create Virtual Environment
        Open command prompt or terminal and run:
            python -m venv venv_311

    Step 3: Activate Virtual Environment
        venv_311\Scripts\activate (Windows)
        venv_311/bin/activate (Linux/Mac)
    
    Step 4: Install Dependencies
        (First, install the required libraries):
        pip install -r requirements.txt
        pip install git+https://github.com/JustAnotherArchivist/snscrape.git

Usage:
    Run the Project
        python main.py


PROJECT STRUCTURE

    PROJ_QALLP
    ├── main.py             # Main scraper script
    ├── requirements.txt    # Python dependencies
    ├── backup_tweets.csv   # Output file (generated after running)
    └── README.md           # This file

Notes:

    - The script uses a free scraping method (no paid APIs required)
    - Rate limiting is built-in to avoid overwhelming the service
    - Data is automatically filtered to the last 24 hours
    - Virtual environment ensures compatibility with snscrape

Troubleshooting:

    If you encounter issues:
        - Ensure Python 3.11.9 is installed correctly
        - Verify virtual environment is activated
        - Check internet connection
        - Make sure all dependencies are installed
