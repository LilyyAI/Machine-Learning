
from flask import Flask, Response, abort
from pathlib import Path

app = Flask(__name__)

CODE_DIR = Path(__file__).parent / "codes"

@app.route("/")
def home():
    return "Machine Learning Practicals - 14 Experiments"

@app.route("/t/<int:number>")
def get_experiment(number):
    if number < 1 or number > 14:
        abort(404)

    file_path = CODE_DIR / f"experiment_{number:02d}.py"

    if not file_path.is_file():
        abort(404, "Experiment code is not uploaded yet.")

    return Response(
        file_path.read_text(encoding="utf-8"),
        mimetype="text/plain"
    )
