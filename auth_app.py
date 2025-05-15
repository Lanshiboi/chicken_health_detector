import sys
import os
import json
from PyQt5 import QtCore, QtWidgets, QtWebChannel
from PyQt5.QtWebEngineWidgets import QWebEngineView
from app import MainApp

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from werkzeug.security import generate_password_hash, check_password_hash

Base = declarative_base()

# Database setup
engine = create_engine('sqlite:///chicken_health.db', echo=False)
Session = sessionmaker(bind=engine)
session = Session()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(String(120), unique=True, nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    password_hash = Column(String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

Base.metadata.create_all(engine)

class BackendBridge(QtCore.QObject):
    loginSuccess = QtCore.pyqtSignal()

    @QtCore.pyqtSlot(str, result='QVariant')
    def register(self, registration_json):
        try:
            data = json.loads(registration_json)
            email = data.get('email')
            first_name = data.get('firstName')
            last_name = data.get('lastName')
            password = data.get('password')
            confirm_password = data.get('confirmPassword')

            if not email or not password or not confirm_password:
                return {'success': False, 'message': 'Email and password are required.'}

            if password != confirm_password:
                return {'success': False, 'message': 'Passwords do not match.'}

            existing_user = session.query(User).filter_by(email=email).first()
            if existing_user:
                return {'success': False, 'message': 'Email already registered.'}

            new_user = User(email=email, first_name=first_name, last_name=last_name)
            new_user.set_password(password)
            session.add(new_user)
            session.commit()

            user_info = {
                'email': new_user.email,
                'firstName': new_user.first_name,
                'lastName': new_user.last_name
            }

            return {'success': True, 'user': user_info, 'token': 'new-token'}

        except Exception as e:
            return {'success': False, 'message': f'Registration failed: {str(e)}'}

    @QtCore.pyqtSlot()
    def notifyLoginSuccess(self):
        print("Login success notified from frontend")
        self.loginSuccess.emit()

class AuthApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Login / Registration - Poultry System')
        self.setGeometry(100, 100, 800, 600)

        self.browser = QWebEngineView()
        html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'auth.html'))

        if os.path.exists(html_path):
            url = QtCore.QUrl.fromLocalFile(html_path.replace("\\", "/"))
            self.browser.setUrl(url)
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "auth.html file not found!")

        self.setCentralWidget(self.browser)

        self.channel = QtWebChannel.QWebChannel()
        self.backend = BackendBridge()
        self.channel.registerObject('backend', self.backend)
        self.browser.page().setWebChannel(self.channel)

        self.backend.loginSuccess.connect(self.onLoginSuccess)

    def onLoginSuccess(self):
        self.close()
        self.main_window = MainApp()
        self.main_window.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AuthApp()
    window.show()
    sys.exit(app.exec_())
