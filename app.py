from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://gentle-cliff-023d4d00f.5.azurestaticapps.net"])

# Register blueprints here
from routes import routes
app.register_blueprint(routes)

@app.route('/')
def index():
    return "Flask server is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
