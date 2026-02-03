# trading/auth.py
import requests
import pyotp
from urllib.parse import urlparse, parse_qs
from kiteconnect import KiteConnect
from django.utils import timezone
import time
import logging

logger = logging.getLogger(__name__)


def generate_and_set_access_token_db(kite: KiteConnect, broker) -> str | None:
    """
    Generate fresh Zerodha access token using automated login + 2FA flow.
    Retries up to 3 times on common failures.
    Returns access_token string on success, None on final failure.
    """
    MAX_ATTEMPTS = 3
    RETRY_DELAY = 15  # seconds

    # Reuse same session across retries for continuity
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info(f"[AUTH ATTEMPT {attempt}/{MAX_ATTEMPTS}] Starting login for {broker.zerodha_user_id}")

        try:
            # Step 1: Load login page
            login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={broker.api_key}"
            resp = session.get(login_url, timeout=15)
            if resp.status_code != 200:
                raise ConnectionError(f"Login page GET failed: {resp.status_code} - {resp.reason}")

            logger.debug(f"[AUTH] Login page loaded (URL: {resp.url})")

            # Step 2: POST username + password
            login_resp = session.post(
                "https://kite.zerodha.com/api/login",
                data={
                    "user_id": broker.zerodha_user_id,
                    "password": broker.password
                },
                timeout=15
            )
            login_data = login_resp.json()

            if login_data.get("status") != "success":
                error_msg = login_data.get('message', login_data)
                raise ValueError(f"Login failed: {error_msg}")

            request_id = login_data["data"]["request_id"]
            logger.debug(f"[AUTH] Login success - request_id: {request_id}")

            # Step 3: POST TOTP (2FA)
            totp_code = pyotp.TOTP(broker.totp).now()
            twofa_resp = session.post(
                "https://kite.zerodha.com/api/twofa",
                data={
                    "user_id": broker.zerodha_user_id,
                    "request_id": request_id,
                    "twofa_value": totp_code,
                    "twofa_type": "totp"
                },
                timeout=15
            )
            twofa_data = twofa_resp.json()

            if twofa_data.get("status") != "success":
                error_msg = twofa_data.get('message', twofa_data)
                raise ValueError(f"2FA failed: {error_msg}")

            logger.debug("[AUTH] 2FA successful")

            # Step 4: Follow redirect to extract request_token
            # Use the original login URL + skip_session to avoid session issues
            final_resp = session.get(login_url + "&skip_session=true", allow_redirects=True, timeout=15)
            final_url = final_resp.url
            logger.debug(f"[AUTH] Redirect final URL: {final_url}")

            # Parse request_token from query string
            parsed = urlparse(final_url)
            query = parse_qs(parsed.query)
            request_token = query.get("request_token", [None])[0]

            if not request_token:
                raise ValueError(f"No request_token found in redirect URL: {final_url}")

            logger.info(f"[AUTH] Request token extracted successfully (len={len(request_token)})")

            # Step 5: Generate Kite session
            data = kite.generate_session(
                request_token=request_token,
                api_secret=broker.secret_key
            )

            access_token = data.get("access_token")
            if not access_token:
                raise ValueError("No access_token returned from generate_session")

            # Step 6: Set token and save to DB
            kite.set_access_token(access_token)

            broker.access_token = access_token
            broker.token_generated_at = timezone.now()
            broker.request_token = None  # clear after successful use
            broker.save(update_fields=['access_token', 'token_generated_at', 'request_token'])

            logger.info(f"[AUTH SUCCESS] Access token generated & saved (prefix: {access_token[:6]}...)")
            return access_token

        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"[AUTH ERROR] Attempt {attempt} failed: {error_type} - {str(e)}")

            if attempt < MAX_ATTEMPTS:
                logger.info(f"[AUTH] Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.critical(f"[AUTH] All {MAX_ATTEMPTS} attempts failed for {broker.zerodha_user_id}")
                return None

    return None