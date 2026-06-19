import sys
import os
import pandas as pd  # noqa: F401 – re-exported for clarity; used in Flask_App/app.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Flask_App'))
from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
