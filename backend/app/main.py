from flask import Flask
from .routes import main_routes

def create_app():
    app = Flask(
        __name__,
        static_folder="../static",
        static_url_path="/static",
        template_folder="../templates"
    )

    # 註冊藍圖
    app.register_blueprint(main_routes)

    return app

if __name__ == "__main__":
    app = create_app()
    # 例如在 localhost:5000 上開啟
    app.run(host="0.0.0.0", port=5000, debug=True)
