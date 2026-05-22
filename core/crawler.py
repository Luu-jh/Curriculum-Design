# core/crawler.py
import time
from datetime import datetime, timedelta

import requests
import urllib3

import config
from core.models import Post, db


class CampusWallCrawler:
    def __init__(self, app):
        self.app = app
        self.today_str = datetime.now().strftime("%Y-%m-%d")
        self.verify_ssl = getattr(config, "CRAWLER_VERIFY_SSL", True)
        if not self.verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def run_incremental_crawl(self, backfill_days=None):
        """执行增量爬取逻辑；传入 backfill_days 时会回填到指定天数边界。"""
        with self.app.app_context():
            is_backfill = backfill_days is not None
            cutoff_time = datetime.now() - timedelta(days=backfill_days) if is_backfill else None
            max_pages = config.STARTUP_BACKFILL_PAGE_COUNT if is_backfill else config.CRAWL_PAGE_COUNT
            mode_name = f"启动回填最近 {backfill_days} 天" if is_backfill else "增量扫描"

            print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始{mode_name}校园墙...")

            total_new = 0
            total_updated = 0
            consecutive_old_count = 0
            break_flag = False

            for page in range(1, max_pages + 1):
                if break_flag:
                    break

                print(f"  正在请求第 {page} 页...")
                try:
                    resp = requests.post(
                        url=config.API_URL,
                        headers=config.HEADERS,
                        json={"page": page, "size": 20},
                        timeout=15,
                        verify=self.verify_ssl,
                    )
                    resp.raise_for_status()

                    data = resp.json()
                    if data.get("code") != "0000":
                        print(f"  接口返回错误: {data.get('message')}")
                        break

                    post_list = data.get("data", {}).get("list", [])
                    if not post_list:
                        print("  已无更多数据。")
                        break

                    for item in post_list:
                        post_id = str(item.get("id"))
                        create_time_str = item.get("create_time", "")
                        create_time = datetime.strptime(create_time_str, "%Y-%m-%d %H:%M:%S")
                        is_top = item.get("is_top") == "1"

                        if is_backfill and create_time < cutoff_time and not is_top:
                            print(f"  已回填到 {backfill_days} 天边界，停止继续翻页。")
                            break_flag = True
                            break

                        existing_post = db.session.get(Post, post_id)

                        if not existing_post:
                            new_post = Post(
                                id=post_id,
                                category=item.get("category_name"),
                                content=item.get("detail", "").replace("\n", " "),
                                author=item.get("show_user_name"),
                                create_time=create_time,
                                views=int(item.get("views", 0)),
                                comments=int(item.get("count_comment", 0)),
                                likes=int(item.get("count_star", 0)),
                            )
                            db.session.add(new_post)
                            total_new += 1
                            consecutive_old_count = 0
                        else:
                            if is_backfill or create_time_str.startswith(self.today_str):
                                existing_post.views = int(item.get("views", 0))
                                existing_post.comments = int(item.get("count_comment", 0))
                                existing_post.likes = int(item.get("count_star", 0))
                                total_updated += 1
                                consecutive_old_count = 0
                            elif not is_top:
                                consecutive_old_count += 1

                        if not is_backfill and consecutive_old_count >= 15:
                            print(f"  已触达历史数据分界线（连续 {consecutive_old_count} 条老帖）。")
                            break_flag = True
                            break

                    db.session.commit()
                    time.sleep(1)

                except requests.exceptions.SSLError as e:
                    print(
                        f"  抓取第 {page} 页失败: SSL证书校验失败。"
                        "如确认接口可信，可在 config.py 中设置 CRAWLER_VERIFY_SSL = False。"
                        f" 原始错误: {e}"
                    )
                    db.session.rollback()
                    break
                except Exception as e:
                    print(f"  抓取第 {page} 页失败: {e}")
                    db.session.rollback()
                    break

            print(f"  本轮结束: 新增 {total_new} 条, 更新热度 {total_updated} 条。")
            self._clean_old_data()

    def _clean_old_data(self):
        """清理 config.DAYS_TO_KEEP 天之前的数据。"""
        cutoff_date = datetime.now() - timedelta(days=config.DAYS_TO_KEEP)
        deleted_count = Post.query.filter(Post.create_time < cutoff_date).delete()
        db.session.commit()
        if deleted_count > 0:
            print(f"  已清理 {deleted_count} 条超过 {config.DAYS_TO_KEEP} 天的历史数据。")
