import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pymysql
import requests
from flask import Flask

import config
from core.models import Post, db


TEST_POST_ID = "__codex_db_test__"


def check_mysql_server():
    print("[1] Checking MySQL server connection...")
    conn = pymysql.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        user=config.DB_USER,
        password=config.DB_PASS,
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{config.DB_NAME}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cursor.execute(f"USE `{config.DB_NAME}`")
            cursor.execute("SHOW TABLES")
            print("    MySQL OK. Existing tables:", [row[0] for row in cursor.fetchall()])
        conn.commit()
    finally:
        conn.close()


def make_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = config.SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = config.SQLALCHEMY_TRACK_MODIFICATIONS
    db.init_app(app)
    return app


def check_sqlalchemy_insert():
    print("[2] Checking SQLAlchemy create_all + insert/read/delete...")
    app = make_app()
    with app.app_context():
        db.create_all()

        old = db.session.get(Post, TEST_POST_ID)
        if old is not None:
            db.session.delete(old)
            db.session.commit()

        post = Post(
            id=TEST_POST_ID,
            category="diagnose",
            content="database diagnostic row",
            author="codex",
            create_time=datetime.now(),
            views=1,
            comments=2,
            likes=3,
        )
        db.session.add(post)
        db.session.commit()

        saved = db.session.get(Post, TEST_POST_ID)
        print(
            "    Insert OK:",
            saved is not None,
            {
                "id": saved.id,
                "category": saved.category,
                "views": saved.views,
                "comments": saved.comments,
                "likes": saved.likes,
            }
            if saved
            else None,
        )

        if saved is not None:
            db.session.delete(saved)
            db.session.commit()
            print("    Cleanup OK.")


def check_api():
    print("[3] Checking API response with configured headers...")
    resp = requests.post(
        config.API_URL,
        headers=config.HEADERS,
        json={"page": 1, "size": 3},
        timeout=15,
    )
    print("    HTTP status:", resp.status_code)
    print("    Content-Type:", resp.headers.get("Content-Type"))
    data = resp.json()
    print("    API code:", data.get("code"))
    print("    API message:", data.get("message"))
    items = data.get("data", {}).get("list", [])
    print("    Item count:", len(items))
    if items:
        first = items[0]
        print("    First item keys:", sorted(first.keys()))
        print(
            "    First item sample:",
            {
                "id": first.get("id"),
                "category_name": first.get("category_name"),
                "create_time": first.get("create_time"),
                "detail_present": bool(first.get("detail")),
            },
        )


def main():
    try:
        check_mysql_server()
        check_sqlalchemy_insert()
        check_api()
        print("[DONE] Database write path and API check completed.")
    except Exception as exc:
        print("[FAILED]", type(exc).__name__, exc)
        raise


if __name__ == "__main__":
    main()
