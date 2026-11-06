# EconoCausal: Troubleshooting & Operational Runbook

**Project:** EconoCausal — Dynamic Pricing via Double Machine Learning  
**Environment:** Windows 10/11, Python 3.10+, PowerShell  

---

## 1. Quick Diagnostic Flowchart

```
                 [Issue Occurred]
                        │
      ┌─────────────────┴─────────────────┐
      ▼                                   ▼
[Environment / Setup]               [Runtime / Model]
  ├─ UTF-16 requirements.txt          ├─ WinError 10061 (Endpoints)
  ├─ Virtual Environment (.venv)      ├─ Positive Price Elasticity
  └─ Slow Cold-Start Subprocesses     └─ Email Dispatch Warnings
```

---

## 2. Common Issues, Root Causes, and Proven Solutions

### Issue 1: `requirements.txt` Encoding Error
- **Symptom:** Viewing or reading `requirements.txt` reports:
  `unsupported mime type text/plain; charset=utf-16le`
- **Root Cause:** In Windows PowerShell, using `pip freeze > requirements.txt` generates a file encoded in UTF-16LE with a Byte Order Mark (BOM).
- **Solution:** Re-encode to standard UTF-8 without BOM using Python:
  ```powershell
  python -c "content = open('requirements.txt', 'r', encoding='utf-16').read(); open('requirements.txt', 'w', encoding='utf-8').write(content)"
  ```
  *(Already resolved in project workspace).*

---

### Issue 2: `[WinError 10061]` in `test_endpoints.py`
- **Symptom:**
  ```
  Testing GET /api/summary ... FAILED: <urlopen error [WinError 10061] No connection could be made because the target machine actively refused it>
  ```
- **Root Cause:** Legacy test script attempted live HTTP requests to `http://127.0.0.1:5000` via `urllib.request` before the Flask development server was launched.
- **Solution:** `test_endpoints.py` has been upgraded to utilize Flask's native WSGI `app.test_client()`, which operates entirely in-memory and does not require an external web server process:
  ```powershell
  .venv\Scripts\python test_endpoints.py
  ```

---

### Issue 3: Cold-Start Import Latency on Windows
- **Symptom:** Commands importing `sklearn`, `scipy`, or `econml` take 30 to 90 seconds on first execution.
- **Root Cause:** The scientific stack contains hundreds of compiled Cython C-extension binaries (`.pyd` files) and OpenBLAS dynamic link libraries. On Windows, the OS loader and security scanner inspect each binary during a cold load.
- **Solution:**
  1. Use unbuffered execution:
     ```powershell
     $env:PYTHONUNBUFFERED="1"
     .venv\Scripts\python -u <script_name.py>
     ```
  2. Once libraries are imported once, Windows retains them in the system page cache, accelerating subsequent executions.

---

### Issue 4: SMTP Email Connection Failures
- **Symptom:** Order placement logs a notice:
  ```
  Notice: Order confirmation email error: [Errno 11001] getaddrinfo failed
  ```
- **Root Cause:** `.env` contains placeholder credentials (`your-email@gmail.com`).
- **Solution:** The `services/email_service.py` module includes automatic safe fallback detection:
  - When placeholder credentials are detected, the system safely routes notifications to the local console log in simulated dev mode without throwing unhandled exceptions.
  - To enable live Gmail delivery, configure `.env` with a real 16-character Gmail App Password:
    ```ini
    MAIL_SERVER=smtp.gmail.com
    MAIL_PORT=587
    MAIL_USERNAME=your-real-email@gmail.com
    MAIL_PASSWORD=your-16-char-app-password
    MAIL_USE_TLS=True
    ```

---

### Issue 5: Naive Model Produces Positive Elasticity ($\theta > 0$)
- **Symptom:** Supervised regression outputs a positive price coefficient, implying raising prices increases sales.
- **Root Cause:** **Price Endogeneity Paradox.** Retail managers systematically raise prices during festive holidays and peak seasons. Standard regression confuses high seasonal demand with price increases.
- **Solution:** Use **Double Machine Learning** (`train_models.py` or `models/dml_model.py`). DML orthogonalizes both demand and price against seasonal confounders using $K$-fold cross-fitting, isolating the microeconomically valid negative causal elasticity ($\theta \approx -0.0152$).

---

### Issue 6: SQLite Database Locked
- **Symptom:** `sqlite3.OperationalError: database is locked`
- **Root Cause:** Concurrently open database connections not closing their cursor or uncommitted transactions.
- **Solution:** All database helpers in `database/database.py` follow the context pattern, ensuring connections and cursors are committed and closed immediately:
  ```python
  conn = get_db_connection()
  try:
      # queries here
      conn.commit()
  finally:
      conn.close()
  ```

---

## 3. Emergency Verification Commands

To verify full system health, run the automated integration test suite:

```powershell
# 1. Test Authentication, Catalog, and E-Commerce
.venv\Scripts\python test_auth_and_ecommerce.py

# 2. Test Order Tracking and Email Fallbacks
.venv\Scripts\python test_email_and_tracking.py

# 3. Test Flask Web Routes & REST APIs
.venv\Scripts\python test_endpoints.py
```
