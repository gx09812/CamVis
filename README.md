# 📷 CamVis

CamVis is a lightweight tool for visualizing camera feeds and running simple analytics (object, face, finger detection).

---

## 📝Features

- 🔴 Real-time camera data visualization  
- 📊 Analytics dashboard and simple reporting  
- 🔌 Easy integration with existing systems  
- ⚙️ Configurable detection options (objects, faces, fingers)

---

## 📤Output

```
    ______              _    ___     
  / ____/___ _____ ___| |  / (_)____
 / /   / __ `/ __ `__ \ | / / / ___/
 / /___/ /_/ / / / / / / |/ / (__  ) 
 \____/\__,_/_/ /_/ /_/|___/_/____/  
                                                 
Usage Example:
    python detect.py --object person car --count 5 --input_source 0 --fingers --face

Options:
    --object       Specify object(s) to detect (e.g. person car dog)
    --count        Number of objects to detect
    --input_source Camera input source (default: 0)
    --fingers      Enable finger detection
    --face         Enable face detection

Press 'q' to quit the video stream
====================================================================================================
```

---

## 🚀 Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/gx09812/CamVis.git CamVis
    cd CamVis
    ```
2. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    # Windows (Command Prompt / PowerShell)
    .venv\Scripts\activate
    # macOS / Linux
    source .venv/bin/activate
    ```
3. Install dependencies (if provided):
    ```bash
    pip install -r requirements.txt
    ```
4. Start the application:
    ```bash
    python CamTool.py
    ```

---

## 🔎Contributing

Contributions welcome — open issues or submit pull requests. Please include a clear description and minimal reproduction steps for any bugfix or feature.

--- 

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](./LICENSE) file for details.
