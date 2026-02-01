import requests
import pyotp
from urllib.parse import urlparse, parse_qs
from kiteconnect import KiteConnect
from django.utils import timezone
import time
import logging

logger = logging.getLogger(__name__)

def generate_and_set_access_token_db(kite: KiteConnect, broker) -> str:
    """Generate fresh access token using automated login flow"""
    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        logger.info(f"[AUTH ATTEMPT {attempt}/{max_attempts}] Starting login for {broker.zerodha_user_id}")

        try:
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })

            login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={broker.api_key}"
            resp = session.get(login_url, timeout=15)
            if resp.status_code != 200:
                raise Exception(f"Login page GET failed: {resp.status_code} - {resp.text[:200]}")

            logger.debug(f"[AUTH] Login page loaded, current URL: {resp.url}")

            # Step 1: POST username + password
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
                raise Exception(f"Login failed: {login_data.get('message', login_data)}")

            request_id = login_data["data"]["request_id"]
            logger.debug(f"[AUTH] Login success, request_id: {request_id}")

            # Step 2: POST TOTP
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
                raise Exception(f"2FA failed: {twofa_data.get('message', twofa_data)}")

            logger.debug("[AUTH] 2FA success")

            # Step 3: Follow redirect to get request_token
            final_resp = session.get(resp.url + "&skip_session=true", allow_redirects=True, timeout=15)
            final_url = final_resp.url
            logger.debug(f"[AUTH] Final redirect URL: {final_url}")

            # Extract request_token
            parsed = urlparse(final_url)
            query = parse_qs(parsed.query)
            request_token = query.get("request_token", [None])[0]

            if not request_token:
                logger.warning(f"[AUTH] No request_token in URL: {final_url}")
                if attempt < max_attempts:
                    logger.info("[AUTH] Retrying login in 15 seconds...")
                    time.sleep(15)
                    continue
                raise Exception(f"Could not extract request_token from redirect URL: {final_url}")

            logger.info(f"[AUTH] Request token extracted: {request_token[:10]}...")

            # Step 4: Generate session
            data = kite.generate_session(request_token=request_token, api_secret=broker.secret_key)
            access_token = data.get("access_token")

            if not access_token:
                raise Exception("No access_token returned from generate_session")

            kite.set_access_token(access_token)

            # Save to DB
            broker.access_token = access_token
            broker.token_generated_at = timezone.now()
            broker.save(update_fields=['access_token', 'token_generated_at'])

            logger.info(f"[AUTH SUCCESS] Access token generated and saved (prefix: {access_token[:6]}...)")
            return access_token

        except Exception as e:
            logger.error(f"[AUTH ERROR] Attempt {attempt} failed: {str(e)}")
            if attempt < max_attempts:
                logger.info("[AUTH] Retrying in 15 seconds...")
                time.sleep(15)
            else:
                raise

    raise Exception("All authentication attempts failed")