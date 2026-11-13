
from flask import Flask, Response, abort
from pathlib import Path

app = Flask(__name__)

CODE_DIR = Path(__file__).parent / "codes"

EXPERIMENTS = {
    1: "experiment_01.py",
    2: "experiment_02_dimensionality_reduction_pca.py",
    3: "experiment_03_lda.py",
    4: "experiment_04_multiple_linear_regression.py",
    5: "experiment_05_polynomial_regression.py",
    6: "experiment_06_ridge_regression.py",
    7: "experiment_07_lasso_regression.py",
    8: "experiment_08_logistic_regression.py",
    9: "experiment_09_linear_svm.py",
    10: "experiment_10_nonlinear_svm.py",
    11: "experiment_11_kmeans.py",
    12: "experiment_12_dbscan.py",
    13: "experiment_13_ensemble_learning.py",
    14: "experiment_14_reinforcement_learning.py"
}

@app.route("/")
def home():
    return "Machine Learning Practicals - 14 Experiments"

@app.route("/t/<int:number>")
def get_experiment(number):
    filename = EXPERIMENTS.get(number)

    if filename is None:
        abort(404)

    file_path = CODE_DIR / filename

    if not file_path.is_file():
        abort(404, "Experiment code is not uploaded yet.")

    return Response(
        file_path.read_text(encoding="utf-8"),
        mimetype="text/plain"
    )
