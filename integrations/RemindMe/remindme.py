"""
WhatsApp Task Reminder System
Monitors a Google Sheet with tasks and sends WhatsApp reminders when deadlines approach
"""

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from twilio.rest import Client
from datetime import datetime, timedelta
import time
import os

class WhatsAppTaskReminder:
    def __init__(self, spreadsheet_url, twilio_account_sid, twilio_auth_token, 
                 twilio_whatsapp_number, recipient_whatsapp_number):
        """
        Initialize the WhatsApp Task Reminder system
        
        Args:
            spreadsheet_url: URL or ID of the Google Sheet
            twilio_account_sid: Twilio Account SID
            twilio_auth_token: Twilio Auth Token
            twilio_whatsapp_number: Twilio WhatsApp number (format: whatsapp:+1234567890)
            recipient_whatsapp_number: Recipient's WhatsApp number (format: whatsapp:+1234567890)
        """
        self.spreadsheet_url = spreadsheet_url
        self.recipient_number = recipient_whatsapp_number
        
        # Initialize Twilio client
        self.twilio_client = Client(twilio_account_sid, twilio_auth_token)
        self.twilio_whatsapp_number = twilio_whatsapp_number
        
        # Initialize Google Sheets client
        self.sheet = None
        self.setup_google_sheets()
        
        # Track sent reminders to avoid duplicates
        self.sent_reminders = set()
    
    def setup_google_sheets(self):
        """Set up Google Sheets API connection"""
        try:
            # Define the scope
            scope = ['https://spreadsheets.google.com/feeds',
                     'https://www.googleapis.com/auth/drive']
            
            # Load credentials from JSON file
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                'credentials.json', scope)
            
            # Authorize the client
            client = gspread.authorize(creds)
            
            # Open the spreadsheet
            self.sheet = client.open_by_url(self.spreadsheet_url).sheet1
            print("✓ Successfully connected to Google Sheets")
        except Exception as e:
            print(f"✗ Error connecting to Google Sheets: {e}")
            raise
    
    def get_tasks(self):
        """
        Fetch all tasks from the Google Sheet
        
        Returns:
            List of dictionaries containing task information
        """
        try:
            # Get all records (assumes first row is header)
            records = self.sheet.get_all_records()
            
            tasks = []
            for record in records:
                # Handle different possible column names (case-insensitive)
                task_id = record.get('task id') or record.get('Task ID') or record.get('task_id')
                task_name = record.get('task name') or record.get('Task Name') or record.get('task_name')
                deadline = record.get('task deadline') or record.get('Task Deadline') or record.get('task_deadline')
                
                if task_id and task_name and deadline:
                    tasks.append({
                        'id': str(task_id),
                        'name': task_name,
                        'deadline': deadline
                    })
            
            return tasks
        except Exception as e:
            print(f"✗ Error fetching tasks: {e}")
            return []
    
    def parse_deadline(self, deadline_str):
        """
        Parse deadline string to datetime object
        Supports multiple date formats
        
        Args:
            deadline_str: String representation of deadline
            
        Returns:
            datetime object or None if parsing fails
        """
        formats = [
            '%Y-%m-%d',           # 2024-12-31
            '%d/%m/%Y',           # 31/12/2024
            '%m/%d/%Y',           # 12/31/2024
            '%d-%m-%Y',           # 31-12-2024
            '%Y-%m-%d %H:%M',     # 2024-12-31 14:30
            '%d/%m/%Y %H:%M',     # 31/12/2024 14:30
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(str(deadline_str).strip(), fmt)
            except ValueError:
                continue
        
        print(f"⚠ Could not parse deadline: {deadline_str}")
        return None
    
    def calculate_time_until_deadline(self, deadline):
        """
        Calculate time remaining until deadline
        
        Args:
            deadline: datetime object
            
        Returns:
            timedelta object representing time until deadline
        """
        now = datetime.now()
        return deadline - now
    
    def should_send_reminder(self, task_id, time_until_deadline):
        """
        Determine if a reminder should be sent based on time until deadline
        
        Args:
            task_id: Unique task identifier
            time_until_deadline: timedelta object
            
        Returns:
            Tuple (should_send, urgency_level)
        """
        # Create unique key for this reminder
        reminder_key = f"{task_id}_{time_until_deadline.days}"
        
        # Check if reminder already sent
        if reminder_key in self.sent_reminders:
            return False, None
        
        days = time_until_deadline.days
        hours = time_until_deadline.seconds // 3600
        
        # Define reminder thresholds
        if days < 0:
            return True, "OVERDUE"
        elif days == 0 and hours <= 2:
            return True, "URGENT (2 hours)"
        elif days == 0:
            return True, "TODAY"
        elif days == 1:
            return True, "TOMORROW"
        elif days <= 3:
            return True, f"{days} DAYS"
        elif days == 7:
            return True, "1 WEEK"
        
        return False, None
    
    def send_whatsapp_message(self, message):
        """
        Send WhatsApp message using Twilio
        
        Args:
            message: Message text to send
            
        Returns:
            True if successful, False otherwise
        """
        try:
            message = self.twilio_client.messages.create(
                body=message,
                from_=self.twilio_whatsapp_number,
                to=self.recipient_number
            )
            print(f"✓ Message sent successfully (SID: {message.sid})")
            return True
        except Exception as e:
            print(f"✗ Error sending WhatsApp message: {e}")
            return False
    
    def format_reminder_message(self, task, urgency):
        """
        Format the reminder message
        
        Args:
            task: Task dictionary
            urgency: Urgency level string
            
        Returns:
            Formatted message string
        """
        icon = "🔴" if urgency == "OVERDUE" else "⚠️" if urgency.startswith("URGENT") else "📅"
        
        message = f"{icon} *TASK REMINDER*\n\n"
        message += f"*Task:* {task['name']}\n"
        message += f"*Deadline:* {task['deadline']}\n"
        message += f"*Status:* {urgency}\n"
        message += f"*Task ID:* {task['id']}"
        
        return message
    
    def check_and_send_reminders(self):
        """
        Main function to check all tasks and send reminders as needed
        """
        print(f"\n{'='*50}")
        print(f"Checking tasks at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
        
        tasks = self.get_tasks()
        print(f"Found {len(tasks)} tasks")
        
        reminders_sent = 0
        
        for task in tasks:
            deadline = self.parse_deadline(task['deadline'])
            
            if deadline:
                time_until = self.calculate_time_until_deadline(deadline)
                should_send, urgency = self.should_send_reminder(task['id'], time_until)
                
                if should_send:
                    message = self.format_reminder_message(task, urgency)
                    
                    if self.send_whatsapp_message(message):
                        # Mark as sent
                        reminder_key = f"{task['id']}_{time_until.days}"
                        self.sent_reminders.add(reminder_key)
                        reminders_sent += 1
                        print(f"  → Sent reminder for: {task['name']} ({urgency})")
                    
                    # Small delay between messages
                    time.sleep(1)
        
        print(f"\nTotal reminders sent: {reminders_sent}")
    
    def run_continuous(self, check_interval_minutes=60):
        """
        Run the reminder system continuously
        
        Args:
            check_interval_minutes: How often to check for reminders (in minutes)
        """
        print(f"\n🚀 Starting WhatsApp Task Reminder System")
        print(f"Checking every {check_interval_minutes} minutes")
        print(f"Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.check_and_send_reminders()
                print(f"\n⏰ Next check in {check_interval_minutes} minutes...")
                time.sleep(check_interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n\n✓ Reminder system stopped")


def main():
    """
    Main function to initialize and run the reminder system
    """
    # Configuration - Replace with your actual values
    SPREADSHEET_URL = "YOUR_GOOGLE_SHEET_URL_HERE"
    TWILIO_ACCOUNT_SID = "YOUR_TWILIO_ACCOUNT_SID"
    TWILIO_AUTH_TOKEN = "YOUR_TWILIO_AUTH_TOKEN"
    TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"  # Twilio Sandbox number
    RECIPIENT_WHATSAPP_NUMBER = "whatsapp:+919876543210"  # Your number
    
    # Or use environment variables (more secure)
    SPREADSHEET_URL = os.getenv('GOOGLE_SHEET_URL', SPREADSHEET_URL)
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', TWILIO_ACCOUNT_SID)
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', TWILIO_AUTH_TOKEN)
    TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER', TWILIO_WHATSAPP_NUMBER)
    RECIPIENT_WHATSAPP_NUMBER = os.getenv('RECIPIENT_WHATSAPP_NUMBER', RECIPIENT_WHATSAPP_NUMBER)
    
    # Initialize and run
    reminder_system = WhatsAppTaskReminder(
        spreadsheet_url=SPREADSHEET_URL,
        twilio_account_sid=TWILIO_ACCOUNT_SID,
        twilio_auth_token=TWILIO_AUTH_TOKEN,
        twilio_whatsapp_number=TWILIO_WHATSAPP_NUMBER,
        recipient_whatsapp_number=RECIPIENT_WHATSAPP_NUMBER
    )
    
    # Run continuously (checks every 60 minutes by default)
    reminder_system.run_continuous(check_interval_minutes=60)


if __name__ == "__main__":
    main()