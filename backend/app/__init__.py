import os
from flask import Flask, render_template
from app.extensions import db


def create_app(config_overrides=None):
    app = Flask(__name__, instance_relative_config=True)

    os.makedirs(app.instance_path, exist_ok=True)
    db_path = os.path.join(app.instance_path, "dropout_risk.db")

    app.config.from_mapping(
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{db_path}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        MODEL_PATH=os.path.join(os.path.dirname(__file__), "ml", "dropout_risk_model.joblib"),
    )
    if config_overrides:
        app.config.update(config_overrides)

    db.init_app(app)

    from app.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    with app.app_context():
        db.create_all()

    @app.get("/")
    def index():
        return render_template("index.html")

    return app
