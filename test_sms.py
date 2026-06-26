import os
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

load_dotenv()

SID = os.getenv("TWILIO_SID")
TOKEN = os.getenv("TWILIO_TOKEN")
FROM = os.getenv("TWILIO_FROM")
TO = os.getenv("TWILIO_TO")

print("Testing Twilio SMS...")

try:
    client = Client(SID, TOKEN)

    message = client.messages.create(
        body="UAV DISASTER ALERT TEST - System working!",
        from_=FROM,
        to=TO
    )

    print("SUCCESS! SID:", message.sid)
    print("Status:", message.status)

except TwilioRestException as e:
    print("TWILIO ERROR:", e.code, "|", e.msg)

except Exception as e:
    print("EXCEPTION:", str(e))