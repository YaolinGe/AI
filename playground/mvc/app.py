# app.py
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Model - Handles data logic
class CounterModel:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1
        return self.count
    
    def get_count(self):
        return self.count

# Controller - Handles requests and ties Model with View
counter_model = CounterModel()

@app.route('/')
def index():
    count = counter_model.get_count()
    return render_template('index.html', count=count)

@app.route('/increment', methods=['POST'])
def increment():
    counter_model.increment()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)