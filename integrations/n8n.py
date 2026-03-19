# stamper/integrations/n8n.py
import requests
import logging

N8N_WEBHOOK_URL = "http://localhost:5678/webhook/stamper-tasks"
logger = logging.getLogger(__name__)

def send_task_to_n8n(task: str, source: str = "stamper") -> bool:
    try:
        response = requests.post(N8N_WEBHOOK_URL, json={
            "task": task,
            "metadata": {"source": source}
        }, timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        logger.warning(f"n8n unreachable: {e}")
        return False  # fails silently, won't crash your app