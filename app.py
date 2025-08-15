from flask import Flask, render_template
from train.train import train

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])  # ✅ Allow POST method
def start_training():
    loss_values, predictions, actuals, plot_paths, mse_value = train()
    return render_template('result.html',
                           loss_values=loss_values,
                           predictions=predictions,
                           actuals=actuals,
                           plot_paths=plot_paths,
                           mse_value=mse_value)

if __name__ == '__main__':
    app.run(debug=True)
