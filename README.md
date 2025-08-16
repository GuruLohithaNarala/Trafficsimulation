
🚦 Traffic Simulation Project  
 📌 Overview  
This project simulates traffic flow in a smart city environment using **graph-based modeling** and **machine learning techniques**. The system leverages **Temporal Graph Neural Networks (TGNNs)** and other AI methods to analyze and forecast traffic conditions in real-time.  
 ✨ Features  
- 🛣️ Road network modeling using graphs  
- ⏱️ Real-time traffic flow prediction  
- 📊 Visualizations (heatmaps, time-series graphs, network maps)  
- ⚡ Temporal Graph Neural Network (TGNN) implementation  
- 🖥️ Interactive dashboard for simulation & results  
- 📂 Configurable datasets for different city layouts  

 🏗️ Project Structure  

traffic-simulation/
│── data/                  # Datasets (real or synthetic traffic data)
│── notebooks/             # Jupyter notebooks for experiments
│── src/                   # Source code
│   ├── models/            # TGNN and ML models
│   ├── preprocessing/     # Data preprocessing scripts
│   ├── visualization/     # Graphs, heatmaps, charts
│   └── app/               # Web dashboard (Flask/Streamlit)
│── results/               # Simulation outputs, plots
│── requirements.txt       # Dependencies
│── README.md              # Project documentation

⚙️ Installation  
1. Clone the repository:  

   git clone https://github.com/yourusername/traffic-simulation.git
   cd traffic-simulation

2. Create a virtual environment:

  bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows

3. Install dependencies:


   pip install -r requirements.txt


▶️ Usage

 Run data preprocessing:

  bash
  python src/preprocessing/prepare_data.py

* Train the TGNN model:

bash
  python src/models/train_tgnn.py

* Launch the dashboard:

bash
  python src/app/app.py


 📊 Example Output

* Heatmap of traffic congestion
* Predicted vs actual traffic flow (time-series)
* Graph visualization of the city road network

 🚀 Tech Stack

* **Python**, **PyTorch**, **DGL / PyTorch Geometric**
* **Flask / Streamlit** (Dashboard)
* **Pandas, NumPy, Matplotlib** (Data & Visualization)

 📚 References

* Research on Temporal Graph Neural Networks
* Smart city traffic datasets (OpenStreetMap, METR-LA, PeMS-BAY)

🏆 Contribution

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

📧 Contact

👤 Your Name
📩 naralagurulohitha@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/gurulohitha-narala-2b84602a2)


