import os.path
import base64
from email.message import EmailMessage
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def get_service():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        service = build("gmail", "v1", credentials=creds)
        return service
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

def send_email(to_email, subject, body, attachment_path=None):
    """Create and send an email message
    Print the returned  message id
    Returns: Message object, including message id
    """
    service = get_service()
    if not service:
        return

    try:
        message = EmailMessage()
        message.set_content(body)
        from email.utils import formataddr
        message["To"] = to_email
        # Explicitly set the sender name and email based on user feedback
        sender_email = "dilmurat.personal@gmail.com"
        sender_name = "David Ahmann"
        message["From"] = formataddr((sender_name, sender_email))
        print(f"DEBUG: Set From header to: {message['From']}")
        message["Subject"] = subject

        if attachment_path:
            import mimetypes
            # Guess the content type based on the file's extension.  Encoding
            # will be ignored, although we should check for simple things like
            # gzip'd files.
            ctype, encoding = mimetypes.guess_type(attachment_path)
            if ctype is None or encoding is not None:
                # No guess could be made, or the file is encoded (compressed), so
                # use a generic bag-of-bits type.
                ctype = 'application/octet-stream'
            maintype, subtype = ctype.split('/', 1)
            
            with open(attachment_path, 'rb') as f:
                file_data = f.read()
                file_name = os.path.basename(attachment_path)
                message.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=file_name)

        # encoded message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {"raw": encoded_message}
        send_message = (
            service.users()
            .messages()
            .send(userId="me", body=create_message)
            .execute()
        )
        print(f'Message Id: {send_message["id"]}')
        return send_message
    except HttpError as error:
        print(f"An error occurred: {error}")
        send_message = None
        return send_message

if __name__ == "__main__":
    # Test execution
    print("Authenticating...")
    service = get_service()
    if service:
        print("Authentication successful! You can now use send_email().")
        # Uncomment to test sending:
        # send_email("recipient@example.com", "Test Subject", "This is a test email from Python.")
