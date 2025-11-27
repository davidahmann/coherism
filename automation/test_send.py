from gmail_client import send_email
import os

to_email = "david.ahmann@cdw.ca"
subject = "Endorsement Request: Coherism: A Variational Feedback Framework for Quantum Information and Spacetime Geometry (Relevant to your work on spacetime foam and QG phenomenology)"
body = """Dear Professor Carlip,

I hope this email finds you well.

I am an independent researcher working on a variational framework for coupling quantum information and spacetime geometry. I have followed your work on spacetime foam and the phenomenology of quantum gravity, and I believe my preprint, "Coherism: A Variational Feedback Framework for Quantum Information and Spacetime Geometry," might be of interest to you.

Because my arXiv account is new, I require a one-time endorsement to submit to the gr-qc category.

I have attached the draft PDF so you can verify that it is a standard mathematical physics manuscript (utilizing variational methods, information geometry, and open quantum systems). I am not asking for a full review, but simply for you to confirm that the paper is appropriate for the subject area.

If you are comfortable providing an endorsement, you can do so via this direct link:
https://arxiv.org/auth/endorse?x=AE8ZU7

Alternatively, you can use the code AE8ZU7 at http://arxiv.org/auth/endorse.php.

Thank you for your time and consideration.

Best regards,

David Ahmann"""

attachment_path = os.path.abspath("../physics/coherism.pdf")

print(f"Sending email to {to_email} with attachment {attachment_path}")
send_email(to_email, subject, body, attachment_path)
