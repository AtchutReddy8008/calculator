from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.contrib.auth.models import User
from django.utils import timezone
import time
import traceback
import logging
import signal
import sys

from .models import Broker, BotStatus
from .core.bot_original import TradingApplication
from .core.auth import generate_and_set_access_token_db

logger = logging.getLogger(__name__)


class BotRunner:
    """Per-task control flag for graceful shutdown"""
    def __init__(self):
        self.running = True

    def stop(self):
        self.running = False


def signal_handler(sig, frame):
    """Graceful shutdown handler"""
    logger.info("Received shutdown signal - stopping bot gracefully")
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


@shared_task(bind=True, time_limit=None, soft_time_limit=None)
def run_user_bot(self, user_id):
    """
    Main Celery task that runs the trading bot for a specific user.
    Includes detailed debug prints, heartbeat persistence, and error handling.
    """
    print(f"[TASK START] run_user_bot launched for user_id={user_id} at {timezone.now()} (task_id={self.request.id})")
    logger.info(f"Bot task started for user_id: {user_id} | Task ID: {self.request.id}")

    runner = BotRunner()
    bot_status = None

    try:
        print("[STEP 1] Loading user from database")
        user = User.objects.get(id=user_id)
        print(f"[STEP 1 OK] User loaded: {user.username} (ID: {user.id})")

        print("[STEP 2] Getting or creating BotStatus record")
        bot_status, created = BotStatus.objects.get_or_create(user=user)
        bot_status.celery_task_id = self.request.id
        bot_status.is_running = True
        bot_status.last_started = timezone.now()
        bot_status.last_error = None
        bot_status.save(update_fields=['celery_task_id', 'is_running', 'last_started', 'last_error'])
        print(f"[STEP 2 OK] BotStatus {'created' if created else 'updated'} and saved")

        print("[STEP 3] Loading Zerodha broker credentials")
        broker = Broker.objects.filter(user=user, broker_name='ZERODHA', is_active=True).first()
        if not broker:
            raise ValueError("No active Zerodha broker found for this user")
        print("[STEP 3 OK] Broker credentials loaded")

        print("[STEP 4] Initializing TradingApplication")
        app = TradingApplication(user=user, broker=broker)
        print("[STEP 4 OK] TradingApplication instance created successfully")

        # Force initial heartbeat right after initialization
        print("[STEP 4.5] Saving forced initial heartbeat")
        bot_status.last_heartbeat = timezone.now()
        bot_status.current_unrealized_pnl = 0
        bot_status.current_margin = 0
        bot_status.save(update_fields=['last_heartbeat', 'current_unrealized_pnl', 'current_margin'])
        print("[STEP 4.5 OK] Initial heartbeat saved → check admin panel now")

        print("[STEP 5] Starting main bot loop (app.run())")
        logger.info(f"Entering main bot loop for user: {user.username}")

        # Heartbeat settings (safety net in task wrapper)
        HEARTBEAT_INTERVAL = 30  # seconds
        last_heartbeat = time.time()

        while runner.running:
            try:
                # Run one full cycle of the bot's internal logic
                app.run()  # This is the real infinite loop from bot_original.py

                # Extra heartbeat (in case app.run() blocks for too long)
                if time.time() - last_heartbeat >= HEARTBEAT_INTERVAL:
                    bot_status.last_heartbeat = timezone.now()
                    try:
                        bot_status.current_unrealized_pnl = float(app.engine.algo_pnl() or 0)
                        bot_status.current_margin = float(app.engine.actual_used_capital() or 0)
                    except Exception as pnl_err:
                        logger.warning(f"Could not update PnL/margin: {pnl_err}")
                    bot_status.save(update_fields=['last_heartbeat', 'current_unrealized_pnl', 'current_margin'])
                    last_heartbeat = time.time()
                    print(f"[HEARTBEAT] Updated BotStatus at {timezone.now()}")

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt in main loop - stopping")
                runner.stop()
            except Exception as loop_err:
                error_msg = f"Error in bot main loop: {str(loop_err)}\n{traceback.format_exc()}"
                print(f"[LOOP ERROR] {error_msg}")
                logger.error(error_msg)
                if bot_status:
                    bot_status.last_error = str(loop_err)[:500]
                    bot_status.save(update_fields=['last_error'])
                time.sleep(10)  # backoff before retry

        logger.info("Main bot loop exited cleanly")

    except SoftTimeLimitExceeded:
        logger.warning("Soft time limit exceeded - shutting down")
        if bot_status:
            bot_status.last_error = "Task soft time limit exceeded"
            bot_status.is_running = False
            bot_status.last_stopped = timezone.now()
            bot_status.save()

    except Exception as fatal_err:
        error_msg = f"Fatal error in run_user_bot: {str(fatal_err)}\n{traceback.format_exc()}"
        print(f"[FATAL CRASH] {error_msg}")
        logger.critical(error_msg)
        if bot_status:
            bot_status.last_error = error_msg[:500]
            bot_status.is_running = False
            bot_status.last_stopped = timezone.now()
            bot_status.save()

    finally:
        print("[CLEANUP] Task finally block - marking bot as stopped")
        if bot_status:
            bot_status.is_running = False
            bot_status.last_stopped = timezone.now()
            bot_status.save()
        logger.info(f"Bot task completed for user_id {user_id}")


@shared_task
def check_bot_health():
    """Periodic Celery Beat task to detect and clean up stale bot records"""
    print("[HEALTH CHECK] Running bot health check")
    now = timezone.now()
    running_bots = BotStatus.objects.filter(is_running=True)

    logger.info(f"Health check found {running_bots.count()} bots marked as running")

    for status in running_bots:
        if status.last_heartbeat:
            age = now - status.last_heartbeat
            if age > timedelta(minutes=5):
                logger.warning(f"Stale bot detected for {status.user.username} — last heartbeat {age}")
                status.is_running = False
                status.last_error = f"Stale (no heartbeat for {age})"
                status.save(update_fields=['is_running', 'last_error'])
        elif status.last_started and (now - status.last_started) > timedelta(minutes=30):
            logger.warning(f"Old bot without heartbeat: {status.user.username}")
            status.is_running = False
            status.last_error = "No heartbeat - assumed dead"
            status.save(update_fields=['is_running', 'last_error'])