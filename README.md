# FYP2026: Raspberry Pi Sound Source Localization System

A real-time sound source localization and visualization system developed for Raspberry Pi.  
This project estimates the direction of a sound source from a microphone array, displays the localization result on a web interface, and optionally uploads results to a cloud server for remote monitoring and historical analysis.

---

## Overview

This project is designed for real-time acoustic source localization with a lightweight deployment workflow. It integrates signal acquisition, localization computation, web visualization, and cloud data uploading into a single system.

The main goals of the project are:

- capture multi-channel audio from a microphone array
- estimate the sound source direction in real time
- display localization results on a browser-based interface
- upload angle, loudness, and timestamp data to a remote server
- support Raspberry Pi deployment and automatic startup

This repository is part of a Final Year Project focused on visualized sound source localization.

---

## Main Features

- **Real-time sound source localization**
  - estimate sound direction from microphone array signals
  - support continuous online localization

- **Browser-based visualization**
  - display localization results in a web page
  - suitable for mobile phone or remote browser access

- **Cloud uploading**
  - upload localization results to a remote server
  - data fields can include angle, loudness, and timestamp

- **Raspberry Pi deployment**
  - designed to run on Raspberry Pi Linux environment
  - includes shell scripts for easier startup

- **Modular code structure**
  - localization logic
  - Flask backend
  - cloud uploader
  - static frontend resources

---

## Project Structure

```bash
.
├── app.py                     # Main Flask application
├── localization.py            # Sound source localization logic
├── cloud_uploader.py          # Cloud upload worker
├── start_localization.sh      # Startup script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── templates/                 # HTML templates
├── static/                    # Frontend static files
│   ├── css/
│   └── js/
├── __pycache__/               # Python cache files
└── other utility scripts      # Experimental / backup / helper scripts
