import smtplib
from email.MIMEText import MIMEText
from email.MIMEMultipart import MIMEMultipart

def notify_by_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = 'wjfustc@gmail.com' 
    msg['To'] = 'jianfengwang@outlook.com' 
    msg['Subject'] = subject 
    
    msg.attach(MIMEText(body, 'plain'))
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    
    server.login("wjfustc@gmail.com", "2006wjf2006")
     
    server.sendmail("wjfustc@gmail.com", "jianfengwang@outlook.com", 
            msg.as_string())
    server.quit()

