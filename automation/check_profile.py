from gmail_client import get_service

service = get_service()
try:
    profile = service.users().getProfile(userId='me').execute()
    print(f"Email Address: {profile['emailAddress']}")
except Exception as e:
    print(f"Error fetching profile: {e}")
