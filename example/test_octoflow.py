from pathlib import Path

from flask import Flask

from octoflow.core.tracker import MLflowExperimentReader
from octoflow.models.base import db

exprs_dir = Path(__file__).parent

app = Flask(__name__)

dbpath = exprs_dir / "mlruns.db"
dbpath.unlink(missing_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + str(dbpath)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()
    MLflowExperimentReader.import_to_db(exprs_dir)
