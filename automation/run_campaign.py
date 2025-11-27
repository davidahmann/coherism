import os
import re
import time
from gmail_client import send_email

def parse_campaign_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Split by separator
    sections = content.split('---')
    
    emails = []
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Skip header/instructions if present (heuristic: check for "To:")
        if "To: `" not in section and "To: " not in section:
            continue

        try:
            # Extract To
            to_match = re.search(r'To: `?([^`\n]+)`?', section)
            if not to_match:
                print(f"Skipping section (no To): {section[:50]}...")
                continue
            to_email = to_match.group(1).strip()
            
            # Extract Subject
            subject_match = re.search(r'Subject: (.+)', section)
            if not subject_match:
                print(f"Skipping section (no Subject): {section[:50]}...")
                continue
            subject = subject_match.group(1).strip()
            
            # Extract Body
            # Body starts after Subject line
            body_start = section.find(subject) + len(subject)
            body = section[body_start:].strip()
            
            emails.append({
                'to': to_email,
                'subject': subject,
                'body': body
            })
        except Exception as e:
            print(f"Error parsing section: {e}")
            continue
            
    return emails

def main():
    campaign_file = "../coherism_outreach.md"
    attachment_path = os.path.abspath("../physics/coherism.pdf")
    
    print(f"Parsing campaign file: {campaign_file}")
    emails = parse_campaign_file(campaign_file)
    print(f"Found {len(emails)} emails to send.")
    
    if not emails:
        print("No emails found. Check the markdown format.")
        return

    print("Attachment:", attachment_path)
    if not os.path.exists(attachment_path):
        print("Error: Attachment not found!")
        return

    confirm = input("Do you want to proceed with sending these emails? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Aborted.")
        return

    for i, email in enumerate(emails):
        print(f"[{i+1}/{len(emails)}] Sending to {email['to']}...")
        # send_email(email['to'], email['subject'], email['body'], attachment_path)
        # Uncomment the line above to actually send. 
        # For safety, I will leave it commented out in this generated script 
        # until the user explicitly asks to run it, or I will use a dry run flag.
        # But the user asked to "execute this campaign".
        # I will uncomment it but add a small delay to avoid rate limits.
        
        send_email(email['to'], email['subject'], email['body'], attachment_path)
        time.sleep(2) # 2 second delay between emails

if __name__ == "__main__":
    main()
