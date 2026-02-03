# trading/tasks.py

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded, TaskRevokedError
from django.contrib.auth.models import User
from django.utils import timezone
import time
import traceback
import logging
import signal
import sys
from datetime import timedelta

from .models import Broker, BotStatus, LogEntry
from .core.bot_original import TradingApplication
from .core.auth import generate_and_set_access_token_db

logger = logging.getLogger(__name__)

# Global flag to ensure signals are registered only once
_signals_registered = False


def register_shutdown_signals():
    """Register global shutdown handlers only once"""
    global _signals_registered
    if _signals_registered:
        return

    def graceful_shutdown(sig, frame):
        logger.info(f"Received shutdown signal ({sig}) - stopping gracefully")
        sys.exit(0)

    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
    _signals_registered = True


# Register signals at module level (once per worker)
register_shutdown_signals()


class BotRunner:
    """Simple per-task control object for graceful shutdown"""
    def __init__(self):
        self.running = True

    def stop(self):
        self.running = False


@shared_task(bind=True, time_limit=None, soft_time_limit=None)
def run_user_bot(self, user_id):
    """
    Main long-running Celery task that runs the trading bot for one user.
    Features: heartbeat, graceful shutdown, revocation detection, detailed logging.
    """
    task_id = self.request.id
    logger.info(f"[TASK START] run_user_bot for user_id={user_id} (task_id={task_id})")
    print(f"[TASK START] run_user_bot launched for user_id={user_id} at {timezone.now()} (task_id={task_id})")

    runner = BotRunner()
    bot_status = None

    try:
        # Step 1: Load user
        logger.debug(f"[{task_id}] Loading user {user_id}")
        user = User.objects.get(id=user_id)
        logger.info(f"[{task_id}] User loaded: {user.username} (ID: {user.id})")

        # Step 2: Get/create BotStatus
        bot_status, created = BotStatus.objects.get_or_create(user=user)
        bot_status.celery_task_id = task_id
        bot_status.is_running = True
        bot_status.last_started = timezone.now()
        bot_status.last_error = None
        bot_status.save(update_fields=['celery_task_id', 'is_running', 'last_started', 'last_error'])
        logger.info(f"[{task_id}] BotStatus {'created' if created else 'updated'}")

        # Step 3: Load broker
        broker = Broker.objects.filter(user=user, broker_name='ZERODHA', is_active=True).first()
        if not broker:
            raise ValueError(f"No active Zerodha broker found for user {user.username}")
        if not broker.access_token:
            raise ValueError(f"Missing access_token for user {user.username}")

        logger.info(f"[{task_id}] Broker loaded successfully")

        # Step 4: Initialize bot application
        app = TradingApplication(user=user, broker=broker)
        logger.info(f"[{task_id}] TradingApplication initialized")

        # Force initial heartbeat
        bot_status.last_heartbeat = timezone.now()
        bot_status.current_unrealized_pnl = 0
        bot_status.current_margin = 0
        bot_status.save(update_fields=['last_heartbeat', 'current_unrealized_pnl', 'current_margin'])
        logger.info(f"[{task_id}] Initial heartbeat saved")

        # Main loop settings
        HEARTBEAT_INTERVAL = 30  # seconds
        last_heartbeat = time.time()

        while runner.running:
            # Check revocation / external stop
            try:
                self.request.revoked()  # Raises TaskRevokedError if revoked
            except TaskRevokedError:
                logger.info(f"[{task_id}] Task revoked externally - shutting down")
                runner.stop()
                break

            # Refresh bot status from DB (in case admin/user stopped it)
            bot_status.refresh_from_db()
            if not bot_status.is_running:
                logger.info(f"[{task_id}] BotStatus.is_running=False → graceful shutdown")
                runner.stop()
                break

            try:
                # Run one full bot cycle
                app.run()  # This contains the real while loop from bot_original.py

                # Extra heartbeat safety net
                if time.time() - last_heartbeat >= HEARTBEAT_INTERVAL:
                    try:
                        bot_status.last_heartbeat = timezone.now()
                        bot_status.current_unrealized_pnl = float(app.engine.algo_pnl() or 0)
                        bot_status.current_margin = float(app.engine.actual_used_capital() or 0)
                        bot_status.save(update_fields=[
                            'last_heartbeat',
                            'current_unrealized_pnl',
                            'current_margin'
                        ])
                        last_heartbeat = time.time()
                        logger.debug(f"[{task_id}] Heartbeat updated")
                    except Exception as hb_err:
                        logger.warning(f"[{task_id}] Heartbeat failed: {hb_err}")

            except KeyboardInterrupt:
                logger.info(f"[{task_id}] KeyboardInterrupt caught - stopping")
                runner.stop()
            except SoftTimeLimitExceeded:
                logger.warning(f"[{task_id}] Soft time limit exceeded - exiting cycle")
                break
            except Exception as loop_err:
                error_msg = f"Error in bot loop: {str(loop_err)}\n{traceback.format_exc()}"
                logger.error(f"[{task_id}] {error_msg}")
                if bot_status:
                    bot_status.last_error = str(loop_err)[:500]
                    bot_status.save(update_fields=['last_error'])
                time.sleep(10)  # backoff

        logger.info(f"[{task_id}] Main loop exited cleanly")

    except Exception as fatal_err:
        error_msg = f"Fatal error: {str(fatal_err)}\n{traceback.format_exc()}"
        logger.critical(f"[{task_id}] {error_msg}")
        print(f"[FATAL] {error_msg}")
        if bot_status:
            bot_status.last_error = str(fatal_err)[:500]
            bot_status.is_running = False
            bot_status.last_stopped = timezone.now()
            bot_status.save()

    finally:
        # Always mark as stopped
        if bot_status:
            bot_status.is_running = False
            bot_status.last_stopped = timezone.now()
            bot_status.save(update_fields=['is_running', 'last_stopped'])
        logger.info(f"[{task_id}] Task cleanup complete - bot marked stopped")


@shared_task
def check_bot_health():
    """
    Periodic Celery Beat task to detect and clean up stale/running bot records.
    Runs every ~5 minutes (configure in celery beat schedule).
    """
    now = timezone.now()
    logger.info("[HEALTH CHECK] Starting bot health check")

    running_bots = BotStatus.objects.filter(is_running=True)

    if not running_bots.exists():
        logger.info("[HEALTH CHECK] No bots currently marked as running")
        return

    logger.info(f"[HEALTH CHECK] Checking {running_bots.count()} running bots")

    for status in running_bots:
        issues = []

        # No heartbeat for >5 min → stale
        if status.last_heartbeat:
            age = now - status.last_heartbeat
            if age > timedelta(minutes=5):
                issues.append(f"No heartbeat for {age.total_seconds()/60:.1f} min")
        # Started long ago but no heartbeat
        elif status.last_started and (now - status.last_started) > timedelta(minutes=30):
            issues.append("Long-running without heartbeat")

        if issues:
            msg = "; ".join(issues)
            logger.warning(f"[HEALTH CHECK] Stale bot detected: {status.user.username} - {msg}")
            status.is_running = False
            status.last_error = f"Stale bot - {msg}"
            status.save(update_fields=['is_running', 'last_error'])
        else:
            logger.debug(f"[HEALTH CHECK] Bot {status.user.username} looks healthy")

    logger.info("[HEALTH CHECK] Completed")