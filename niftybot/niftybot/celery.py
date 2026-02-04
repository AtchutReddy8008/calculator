# niftybot/niftybot/celery.py

import os
from celery import Celery
from celery.schedules import crontab

# ------------------------------------------------------------------------------
# SET DJANGO SETTINGS
# ------------------------------------------------------------------------------
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "niftybot.settings")

# ------------------------------------------------------------------------------
# CREATE CELERY APP
# ------------------------------------------------------------------------------
app = Celery("niftybot")

# ------------------------------------------------------------------------------
# LOAD DJANGO CELERY SETTINGS
# ------------------------------------------------------------------------------
app.config_from_object("django.conf:settings", namespace="CELERY")

# ------------------------------------------------------------------------------
# AUTO DISCOVER TASKS
# ------------------------------------------------------------------------------
app.autodiscover_tasks()

# ------------------------------------------------------------------------------
# 🔥 CRITICAL CONFIG — SAFE FOR LONG-RUNNING TRADING BOTS
# ------------------------------------------------------------------------------
app.conf.update(

    # ❌ DO NOT EVER SET TIME LIMITS FOR TRADING BOTS
    task_time_limit=None,
    task_soft_time_limit=None,

    # ✅ Worker behavior (LONG TASK SAFE)
    worker_prefetch_multiplier=1,   # Prevent task stealing
    task_acks_late=True,            # Ack only after clean exit
    worker_concurrency=1,           # ONE BOT PER WORKER (MANDATORY)

    # ✅ Stability
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=None,

    # ✅ Result backend cleanup
    result_expires=3600,

    # ✅ Timezone (Indian market)
    timezone="Asia/Kolkata",
    enable_utc=False,

    # ✅ Visibility
    task_track_started=True,
)

# ------------------------------------------------------------------------------
# CELERY BEAT SCHEDULE
# ------------------------------------------------------------------------------
app.conf.beat_schedule = {

    # Health check every 5 minutes
    "check-bot-health-every-5-minutes": {
        "task": "trading.tasks.check_bot_health",
        "schedule": crontab(minute="*/5"),
        "options": {
            "expires": 300,
        },
    },
}

# ------------------------------------------------------------------------------
# DEBUG TASK (OPTIONAL BUT USEFUL)
# ------------------------------------------------------------------------------
@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f"Celery Debug Request: {self.request!r}")

# ------------------------------------------------------------------------------
# ENTRY POINT (OPTIONAL)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.start()


# # niftybot/niftybot/celery.py
# import os
# from celery import Celery
# from celery.schedules import crontab
# from django.conf import settings

# # Set default Django settings module for Celery
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'niftybot.settings')

# # Create Celery app instance
# app = Celery('niftybot')

# # Load configuration from Django settings with CELERY_ namespace
# app.config_from_object('django.conf:settings', namespace='CELERY')

# # Automatically discover tasks in all installed apps
# # This will find trading.tasks automatically
# app.autodiscover_tasks()

# # Optional: Debug task (very useful during development)
# @app.task(bind=True, ignore_result=True)
# def debug_task(self):
#     print(f'Request: {self.request!r}')
#     return 'Debug task executed successfully'


# # Periodic tasks schedule (Celery Beat)
# app.conf.beat_schedule = {
#     # Run bot health check every 5 minutes
#     'check-bot-health-every-5-minutes': {
#         'task': 'trading.tasks.check_bot_health',
#         'schedule': crontab(minute='*/5'),  # every 5 minutes
#         'args': (),
#         'options': {
#             'expires': 300,  # task result expires after 5 min
#         },
#     },

#     # Optional: Example of another periodic task (uncomment if needed)
#     # 'cleanup-old-logs-every-day': {
#     #     'task': 'trading.tasks.cleanup_old_logs',
#     #     'schedule': crontab(hour=2, minute=0),  # 2:00 AM daily
#     # },
# }

# # Important settings (you can override these in settings.py too)
# app.conf.update(
#     # Result backend (recommended: use Redis or database)
#     result_expires=3600,                # 1 hour
#     task_track_started=True,
#     task_time_limit=7200,               # 2 hours hard limit per task
#     task_soft_time_limit=7000,          # soft limit to allow graceful shutdown
#     worker_concurrency=4,               # adjust based on your server
#     worker_prefetch_multiplier=1,       # better for long-running tasks
#     broker_connection_retry_on_startup=True,
#     broker_connection_max_retries=None,
#     timezone='Asia/Kolkata',            # important for Indian market hours
#     enable_utc=False,
# )

# # Optional: Load task modules explicitly (if autodiscover misses something)
# app.conf.task_modules = [
#     'trading.tasks',
#     # Add more if you create new apps with tasks
# ]

# # This line is required for Django to find the app
# @app.task(bind=True)
# def debug_task(self):
#     print(f'Request: {self.request!r}')


# # Make sure the app is importable
# if __name__ == '__main__':
#     app.start()