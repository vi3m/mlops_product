from flask import Flask, request, jsonify
import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

app = Flask(__name__)

@app.route('/drift-report', methods=['POST'])
def drift_report():
    try:
        data = request.files['file']
        df = pd.read_csv(data)
        
        # Example: Compare the first 50% as reference and the last 50% as current
        n_rows = len(df)
        reference = df[: n_rows // 2]
        current = df[n_rows // 2 :]

        # Generate drift report
        dashboard = Dashboard(tabs=[DataDriftTab()])
        dashboard.calculate(reference, current)
        dashboard.save("drift_report.html")

        return jsonify({"message": "Drift report generated.", "file": "drift_report.html"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8502)
