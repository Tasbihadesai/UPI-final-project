import os
import json
import threading
import smtplib
from email.mime.text import MIMEText

# Settings are read from alert_config.json (same folder as this file).
# If a value is missing there, the matching environment variable is used.
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alert_config.json")
_SEND_LOCK = threading.Lock()


def _serialized(fn):
    """Run phone-app automations one at a time (several messages are sent per transaction)."""
    def wrapper(*args, **kwargs):
        import time
        with _SEND_LOCK:
            fn(*args, **kwargs)
            time.sleep(4)
    return wrapper


_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alert_log.txt")


def _log(msg: str):
    """Writes a line to alert_log.txt (next to this file) so failures are visible."""
    try:
        from datetime import datetime
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")
    except Exception:
        pass


_PLACEHOLDERS = {"yourgmail@gmail.com", "abcdefghijklmnop", "abcd efgh ijkl mnop", "your real gmail address"}


def log_event(msg: str):
    """Public helper: write a line to alert_log.txt (nothing is shown on screen)."""
    _log(msg)


def _use_phone_apps() -> bool:
    """Opening WhatsApp / Messages on this laptop is OFF unless USE_PHONE_APPS is true."""
    return _setting("USE_PHONE_APPS").lower() in ("true", "1", "yes")


def _setting(name: str) -> str:
    """Environment variable wins; otherwise alert_config.json. Placeholder text is ignored."""
    value = os.getenv(name, "").strip()
    if value and value.lower() not in _PLACEHOLDERS:
        return value
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8-sig") as f:
            value = str(json.load(f).get(name, "") or "").strip()
    except Exception:
        value = ""
    if value.lower() in _PLACEHOLDERS:
        return ""
    return value


def build_alert_message(sender, receiver, amount, fraud_prob_pct, status, external=False):
    """Message for the SENDER."""
    note = " (not an ALVAN user)" if external else ""
    return (
        f"ALVAN ALERT: {sender}, you are transferring Rs. {amount:,.2f} to {receiver}{note}. "
        f"Fraud risk detected: {fraud_prob_pct:.2f}%. Status: {status}. "
        f"If this wasn't you, request a recovery immediately."
    )


def build_receiver_message(sender, receiver, amount):
    """Message for the RECEIVER."""
    return (
        f"ALVAN: Hi {receiver}, you have received Rs. {amount:,.2f} from {sender}."
    )


def send_email(to_email, subject, body):
    user = _setting("ALERT_EMAIL_USER")
    pwd = _setting("ALERT_EMAIL_APP_PASS").replace(" ", "")
    if not user:
        return False, f"email not configured: ALERT_EMAIL_USER missing (put it in {_CONFIG_PATH})"
    if not pwd:
        return False, f"email not configured: ALERT_EMAIL_APP_PASS missing (put it in {_CONFIG_PATH})"
    if not to_email:
        return False, "no email saved for this user (save it in 'Alert contact details')"
    m = MIMEText(body, _charset="utf-8")
    m["Subject"], m["From"], m["To"] = subject, user, to_email

    errors = []
    # Try port 587 (STARTTLS) first, then port 465 (SSL).
    for mode, port in (("starttls", 587), ("ssl", 465)):
        try:
            if mode == "starttls":
                with smtplib.SMTP("smtp.gmail.com", port, timeout=25) as s:
                    s.ehlo()
                    s.starttls()
                    s.ehlo()
                    s.login(user, pwd)
                    s.send_message(m)
            else:
                with smtplib.SMTP_SSL("smtp.gmail.com", port, timeout=25) as s:
                    s.login(user, pwd)
                    s.send_message(m)
            return True, f"email sent to {to_email}"
        except smtplib.SMTPAuthenticationError:
            return False, "email failed: Gmail rejected the login. Use a 16-letter App Password (not your Gmail password) and make sure 2-Step Verification is on."
        except Exception as e:
            errors.append(f"port {port}: {e}")

    return False, (
        "email failed on both Gmail ports (" + " | ".join(errors) + "). "
        "Your network, antivirus or firewall is probably blocking SMTP - "
        "try a mobile hotspot or turn off your antivirus 'email shield'."
    )


@_serialized
def _send_sms_via_messages_app(digits: str, body: str):
    """Opens the Windows Messages app (Phone Link) through an sms: link and presses Send."""
    try:
        import time
        import urllib.parse
        import pyautogui
        url = f"sms:+91{digits}?body={urllib.parse.quote(body)}"
        _log(f"sms-app: opening Messages app for {digits}")
        os.startfile(url)
        time.sleep(12)
        pyautogui.press("enter")
        _log("sms-app: pressed Enter (message should be sent)")
    except Exception as e:
        _log(f"sms-app: FAILED - {e}")
        print("SMS via Messages app failed:", e)


def send_sms(to_phone, body):
    """Real SMS to the phone's normal Messages app, through Fast2SMS (India)."""
    digits = "".join(ch for ch in str(to_phone or "") if ch.isdigit())[-10:]
    if len(digits) != 10:
        return False, "no valid 10-digit phone saved for this user (save it in 'Alert contact details')"
    key = _setting("FAST2SMS_KEY")
    if not key and not _use_phone_apps():
        _log("sms: not sent - FAST2SMS_KEY missing (apps are disabled)")
        return False, "sms not sent: add FAST2SMS_KEY in alert_config.json"
    if not key:
        # Optional: use the Messages app (Phone Link) on this laptop
        if not hasattr(os, "startfile"):
            return False, "sms via Messages app works only on Windows"
        try:
            import pyautogui  # noqa: F401
        except ImportError:
            return False, "sms skipped: run  pip install pyautogui"
        import threading
        threading.Thread(target=_send_sms_via_messages_app, args=(digits, body), daemon=True).start()
        return True, (
            f"sms is being sent to {digits} through the Windows Messages app (Phone Link) - "
            "it needs your phone paired in Phone Link; don't touch the keyboard for ~15 seconds"
        )
    try:
        import requests
        r = requests.post(
            "https://www.fast2sms.com/dev/bulkV2",
            headers={"authorization": key},
            data={"route": "q", "message": body, "numbers": digits, "flash": "0"},
            timeout=20,
        )
        data = r.json()
        _log(f"sms: {r.status_code} {data}")
        if data.get("return") is True:
            return True, f"sms sent to {digits}"
        return False, f"sms failed: {data.get('message', data)}"
    except Exception as e:
        _log(f"sms: FAILED - {e}")
        return False, f"sms failed: {e}"


@_serialized
def _send_via_whatsapp_desktop(digits: str, body: str):
    """Uses the installed WhatsApp Windows app directly (whatsapp:// link) - no browser."""
    try:
        import time
        import urllib.parse
        import pyautogui
        url = f"whatsapp://send?phone=91{digits}&text={urllib.parse.quote(body)}"
        _log(f"desktop: opening WhatsApp chat for {digits}")
        os.startfile(url)          # opens WhatsApp desktop with the chat + message ready
        time.sleep(12)             # give the app time to open and load the chat
        pyautogui.press("enter")   # press Send
        _log("desktop: pressed Enter (message should be sent)")
    except Exception as e:
        _log(f"desktop: FAILED - {e}")
        print("WhatsApp desktop send failed:", e)


@_serialized
def _send_via_whatsapp_web(digits: str, body: str):
    """Fallback: WhatsApp Web in the browser (needs pywhatkit)."""
    try:
        import pywhatkit
        pywhatkit.sendwhatmsg_instantly(
            f"+91{digits}", body, wait_time=20, tab_close=True, close_time=5
        )
    except Exception as e:
        print("WhatsApp Web send failed:", e)


def send_whatsapp(to_phone, body):
    digits = "".join(ch for ch in str(to_phone or "") if ch.isdigit())[-10:]
    if len(digits) != 10:
        return False, "no valid 10-digit phone saved for this user (save it in 'Alert contact details')"

    sid = _setting("TWILIO_SID")
    token = _setting("TWILIO_TOKEN")

    # Option A: Twilio (only if you filled in the keys)
    if sid and token:
        try:
            from twilio.rest import Client
            Client(sid, token).messages.create(
                from_="whatsapp:+14155238886",
                to=f"whatsapp:+91{digits}",
                body=body,
            )
            return True, f"whatsapp sent to {digits}"
        except ImportError:
            return False, "whatsapp failed: run  pip install twilio"
        except Exception as e:
            return False, f"whatsapp failed: {e}"

    if not _use_phone_apps():
        _log("whatsapp: not sent - Twilio keys missing (apps are disabled)")
        return False, "whatsapp not sent: add TWILIO_SID and TWILIO_TOKEN in alert_config.json"

    import threading
    mode = (_setting("WHATSAPP_MODE") or "desktop").lower()

    # Option B: WhatsApp desktop app (default)
    if mode == "desktop":
        if not hasattr(os, "startfile"):
            return False, "whatsapp desktop mode works only on Windows"
        try:
            import pyautogui  # noqa: F401
        except ImportError:
            return False, "whatsapp skipped: run  pip install pyautogui"
        threading.Thread(target=_send_via_whatsapp_desktop, args=(digits, body), daemon=True).start()
        return True, (
            f"whatsapp is being sent to {digits} through the WhatsApp desktop app - "
            "it will open and send by itself in ~10 seconds; don't click or type meanwhile"
        )

    # Option C: WhatsApp Web in the browser
    try:
        import pywhatkit  # noqa: F401
    except ImportError:
        return False, "whatsapp skipped: run  pip install pywhatkit"
    threading.Thread(target=_send_via_whatsapp_web, args=(digits, body), daemon=True).start()
    return True, (
        f"whatsapp is being sent to {digits} through WhatsApp Web - a browser tab will open; "
        "don't touch the keyboard for ~25 seconds"
    )


if __name__ == "__main__":
    # Test (no email):   python notifier.py 98XXXXXXXX
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage:  python notifier.py 98XXXXXXXX     (your 10-digit mobile number)")
        sys.exit(0)

    phone = sys.argv[1]
    msg = build_alert_message("TestUser", "6363349820", 1500.0, 12.5, "APPROVED")
    print("SMS     :", send_sms(phone, msg))
    print("WHATSAPP:", send_whatsapp(phone, msg))
    print("Waiting 40 seconds for the apps to open and send - don't touch the keyboard...")
    time.sleep(40)
    print("Done. Details are in:", _LOG_PATH)