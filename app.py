from flask import Flask
from flask_cors import CORS

app = Flask(__name__)  # Creating a Flask application instance
CORS(app)  # Enable CORS for all routes

# Register blueprints here
from routes import routes  # Importing the 'routes' blueprint from routes.py
app.register_blueprint(routes)  # Registering the 'routes' blueprint with the Flask application

@app.route('/')
def index():
    return "Flask server is runningg!"  # A simple route to check if the server is running

if __name__ == '__main__':
    # Running the Flask application on host '0.0.0.0' and port 80
    app.run(host='0.0.0.0', port=80, debug=True)
