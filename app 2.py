from flask import Flask
from flask_cors import CORS  # type: ignore
from routes import configure_routes  # Import dari routes

app = Flask(__name__)
CORS(app)
# Konfigurasi route
configure_routes(app)

if __name__ == "__main__":
    app.run(debug=True)
