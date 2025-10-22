# SolarDoc - Solar Panel Health Analysis

**SolarDoc** is an AI-powered system for detecting defects in solar panels. It uses deep learning with a **ResNet50-based classifier** and transfer learning to identify common defects in solar panel images.

---

## 🌞 Features

- Detects **5 types of solar panel defects**:
  - `broken`
  - `bright_spot`
  - `black_border`
  - `scratched`
  - `non_electricity`
- Pre-trained **ResNet50** with transfer learning for fast training
- Supports **checkpointing** to resume training
- Generates **training plots**, **confusion matrices**, and **prediction visualizations**
- Provides a Python interface for **predicting new images**
- Saves models and results, optionally to **Google Drive**

---

## 📦 Repository Contents

