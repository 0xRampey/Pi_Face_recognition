# Face recognition on the Pi Zero
Run one-shot face recognition on your Raspberry Pi Zero with the help of the Movidius Neural Compute Stick.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Hardware Prerequisites
- 1 NCS device
- A webcam supported by the Pi
- Button (optional)

### Software Prerequisites
First you need to install requirements for the following software,
- dlib and face_recognition (https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65)
- NCS API (https://movidius.github.io/blog/ncs-apps-on-rpi/)

Now, clone the repo into your Pi and run video_face_matcher.py. That's it!
```
git clone https://github.com/prampey/Pi_Face_recognition.git
cd Pi_Face_recognition
python3 video_face_matcher.py 
```

