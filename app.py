"""Entry point per Azure App Service.

App Service cerca per default un oggetto `app` esportato da `app.py` nella
root del progetto. Questo file fa da semplice ponte verso il vero codice in
`backend/main.py`.
"""

from backend.main import app

__all__ = ["app"]
