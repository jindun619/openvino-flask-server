# OpenVINO Flask Server

This project is part of the **校企项目 (Industry-Academia Collaboration Project)** between **Shanghai Jiao Tong University** and **Intel**. It is still under development and incomplete. The project involves building a **Flask server** that operates in a cloud computing environment (e.g., AWS EC2, Featurize, or other GPU-equipped environments). The server processes image requests, generates descriptive text outputs using the **LLaVA model** and prompt-based techniques, and accelerates inference time using **Intel's OpenVINO**. The ultimate goal of the project is to create a system for visually impaired individuals that detects objects on small devices and describes the situation through audio. This is a **team project**, and I am responsible for server development.

## 📚 Tech Stack

![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)![OpenVINO](https://img.shields.io/badge/OpenVINO-FF6600?style=for-the-badge&logo=intel&logoColor=white)![LLaVA](https://img.shields.io/badge/LLaVA-000000?style=for-the-badge)![AWS EC2](https://img.shields.io/badge/AWS_EC2-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)

## 💡 Project Overview

### **Server Functionality**

- Receives image requests from clients.
- Uses the **LLaVA model** (currently `deepseek-ai/deepseek-vl-7b-chat` from [Hugging Face](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)) to generate descriptive text outputs based on the image and provided prompts.
- Accelerates inference time using **Intel's OpenVINO**.
- Returns the generated text as a response.

### **Ultimate Goal**

- Develop a system for visually impaired individuals that:
  - Detects objects using small devices.
  - Describes the situation through audio.

### **Current Progress**

- The server is operational in a **Linux (Ubuntu)** environment.
- The LLaVA model is integrated, but the model may change in the future.
- Focus is on optimizing inference time using **OpenVINO**.

## Key Learnings

- Developed a Flask server for image processing and text generation.
- Integrated the LLaVA model for generating descriptive text outputs.
- Utilized Intel's OpenVINO to optimize inference time.
- Gained experience in deploying and managing servers in cloud environments (e.g., AWS EC2, Featurize).
- Collaborated in a 3-person team for an industry-academia project.

## Future Plans

- Optimize the server further for faster inference and lower latency.
- Explore alternative models for better accuracy and performance.
- Integrate the server with small devices for real-time object detection and audio description.
