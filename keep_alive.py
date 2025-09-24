from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "🤖 Forex Trading Bot is RUNNING 24/7!"

@app.route('/health')
def health():
    return {"status": "active", "message": "Bot is healthy"}

def run_flask(port=5000):
    app.run(host='0.0.0.0', port=port)