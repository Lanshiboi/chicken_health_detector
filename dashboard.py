import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets, QtWebChannel
from PyQt5.QtWebEngineWidgets import QWebEngineView
import json

class BackendBridge(QtCore.QObject):
    @QtCore.pyqtSlot(result='QVariant')
    def get_dashboard_data(self):
        # For demonstration, return mock data
        # TODO: Replace with actual counts from dataset or database
        healthy_count = 10
        possible_fever_count = 3
        possible_bird_flu_count = 2
        infected_count = 3
        at_risk_count = possible_fever_count + possible_bird_flu_count

        data = {
            'statistics': {
                'healthy': healthy_count,
                'atRisk': at_risk_count,
                'infected': infected_count,
                'total': healthy_count + at_risk_count + infected_count
            },
            'recentAlerts': [
                {'date': '2024-06-01', 'chickenId': 'CHK_001', 'status': 'Infected', 'id': '1'},
                {'date': '2024-06-02', 'chickenId': 'CHK_002', 'status': 'At Risk', 'id': '2'},
                {'date': '2024-06-03', 'chickenId': 'CHK_003', 'status': 'Infected', 'id': '3'}
            ],
            'healthOverview': {
                'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                'healthy': [2, 3, 2, 4, 3, 2, 1],
                'infected': [0, 1, 0, 1, 0, 1, 0]
            },
            'distribution': {
                'healthy': healthy_count,
                'atRisk': at_risk_count,
                'infected': infected_count
            }
        }
        return data

    @QtCore.pyqtSlot(str, result='QVariant')
    def show_details(self, alert_id):
        # For demonstration, return mock details for alert
        details = {
            '1': {'info': 'Details for alert 1'},
            '2': {'info': 'Details for alert 2'},
            '3': {'info': 'Details for alert 3'}
        }
        return details.get(alert_id, {'info': 'No details found'})

class DashboardApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('BCHD - Poultry Disease Monitoring Dashboard')
        self.setGeometry(100, 100, 1200, 800)

        self.browser = QWebEngineView()
        html_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dashboard.html'))
        if os.path.exists(html_file_path):
            self.browser.setUrl(QtCore.QUrl.fromLocalFile(html_file_path))
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "dashboard.html file not found!")

        self.setCentralWidget(self.browser)

        self.channel = QtWebChannel.QWebChannel()
        self.backend = BackendBridge()
        self.channel.registerObject('backend', self.backend)
        self.browser.page().setWebChannel(self.channel)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = DashboardApp()
    window.show()
    sys.exit(app.exec_())
