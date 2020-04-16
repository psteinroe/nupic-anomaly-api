#!flask/bin/python
from datetime import datetime
from flask import Flask, request, jsonify

from NupicDetector import NupicDetector

app = Flask(__name__)

detector = None


@app.route('/api/init', methods=['POST'])
def init():
    global detector
    detector = NupicDetector(
        inputMin=request.json['inputMin'],
        inputMax=request.json['inputMax'],
        probationaryPeriod=request.json['probationaryPeriod']
    )
    detector.initialize()
    return "OK"


@app.route('/api/handleRecord', methods=['POST'])
def handle_record():
    global detector
    input_record = {
        'timestamp': datetime.utcfromtimestamp(request.json['timestamp']),
        'value': request.json['value']
    }
    final_score, raw_score = detector.handleRecord(input_record)
    final_score = float(final_score)
    raw_score = float(raw_score)
    return {
        'anomalyScore': final_score,
        'rawScore': raw_score
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
