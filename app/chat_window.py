from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.api_client import ChatClient
from app.chat_worker import ChatWorker


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.client = ChatClient()
        self.worker = None

        self.setWindowTitle("AIチャット")
        self.setMinimumSize(600, 400)

        # チャット表示エリア
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)

        # 入力エリア
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("メッセージを入力...")
        self.input_field.returnPressed.connect(self.send_message)

        self.send_button = QPushButton("送信")
        self.send_button.clicked.connect(self.send_message)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)

        # メインレイアウト
        layout = QVBoxLayout()
        layout.addWidget(self.chat_display)
        layout.addLayout(input_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def send_message(self):
        message = self.input_field.text().strip()
        if not message:
            return

        self.chat_display.append(f"あなた: {message}")
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.send_button.setText("考え中...")

        self.worker = ChatWorker(self.client, message)
        self.worker.response_ready.connect(self.on_response)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.status_update.connect(self.on_status_update)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_response(self, response: str):
        self.chat_display.append(f"AI: {response}\n")

    def on_error(self, error: str):
        self.chat_display.append(f"[エラー] {error}\n")

    def on_status_update(self, status: str):
        self.send_button.setText(status)
        self.chat_display.append(f"[{status}]")

    def on_finished(self):
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.send_button.setText("送信")
        self.input_field.setFocus()
