from PySide6.QtCore import QThread, Signal

from app.api_client import ChatClient


class ChatWorker(QThread):
    response_ready = Signal(str)
    error_occurred = Signal(str)
    status_update = Signal(str)

    def __init__(self, client: ChatClient, message: str):
        super().__init__()
        self.client = client
        self.message = message

    def run(self):
        try:
            response = self.client.send_message_with_tools(
                self.message,
                status_callback=self.status_update.emit,
            )
            self.response_ready.emit(response)
        except Exception as e:
            self.error_occurred.emit(str(e))
