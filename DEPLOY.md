# 🚀 Deploy to PythonAnywhere — Easy Steps

This guide walks you through deploying the HSC Subject Recommendation Engine to PythonAnywhere.

**Your account:** https://www.pythonanywhere.com/user/hscmatch/
**Your live URL (after setup):** https://hscmatch.pythonanywhere.com/

---

## ✅ Prerequisites

- A free or paid PythonAnywhere account (you have: `hscmatch`)
- The latest code pushed to GitHub: https://github.com/Richard-Johnson-Anglican-College/12SE_HSC_subject_selection

---

## 📋 First-Time Deployment (Do This Once)

### **Step 1: Open a Bash Console**

1. Go to https://www.pythonanywhere.com/user/hscmatch/
2. Click **"Consoles"** tab → click **"Bash"** under "New console"

### **Step 2: Clone the Repository**

In the Bash console, run:

```bash
cd ~
git clone https://github.com/Richard-Johnson-Anglican-College/12SE_HSC_subject_selection.git
cd 12SE_HSC_subject_selection
```

> **📝 Note:** Git identity config is only needed if you plan to **commit** changes from PythonAnywhere. For normal deployment (pulling changes from GitHub), no git config is required.

### **Step 3: Create a Virtual Environment** ⚡ Fast Method

```bash
mkvirtualenv --python=/usr/bin/python3.13 --system-site-packages hscmatch-venv
```

This activates automatically. Your prompt will now show `(hscmatch-venv)`.

> **💡 Why `--system-site-packages`?**
> PythonAnywhere already has the exact versions of `numpy`, `pandas`, `matplotlib`, and `scikit-learn` that we need (verified pre-installed in `/usr/lib/python3.13/site-packages`). This flag inherits them, so you skip ~10-20 minutes of slow source compilation on the free tier.

### **Step 4: Install Flask and google-generativeai**

```bash
cd ~/12SE_HSC_subject_selection
pip install Flask==3.0.3 google-generativeai
```

Takes ~30 seconds. The other packages (`scikit-learn`, `pandas`, `numpy`, `matplotlib`) are inherited from the system — no compilation needed.

> **🤖 Note on AI/LLM packages:** If your project uses an AI/LLM library (like `google-generativeai`, `openai`, `anthropic`, etc.), these are **NOT** pre-installed on PythonAnywhere and must be explicitly installed. Only the core data science packages (numpy, pandas, matplotlib, scikit-learn) are pre-installed system-wide.

### **Step 4b: Verify All Packages Are Available**

```bash
python -c "import flask, pandas, numpy, matplotlib, sklearn; print('Flask:', flask.__version__); print('pandas:', pandas.__version__); print('numpy:', numpy.__version__); print('matplotlib:', matplotlib.__version__); print('sklearn:', sklearn.__version__)"
```

You should see:
```
Flask: 3.0.3
pandas: 2.2.2
numpy: 2.1.0
matplotlib: 3.9.2
sklearn: 1.6.0
```

✅ All versions match — perfect.

> **⚠️ Avoid this slow alternative:** Running `pip install -r requirements.txt` in a fresh venv (without `--system-site-packages`) will try to compile `pandas 2.2.2` from source because no Python 3.13 wheel exists. This can hang for 10-20 minutes on the free tier or fail entirely. Always use the `--system-site-packages` approach above.

### **Step 5: Create the Web App**

1. Go to the **"Web"** tab on the PythonAnywhere dashboard
2. Click **"Add a new web app"**
3. Click **"Next"** (the domain `hscmatch.pythonanywhere.com` is auto-selected)
4. Choose **"Manual configuration"** (NOT Flask — we'll configure it manually for control)
5. Choose **"Python 3.13"**
6. Click **"Next"** to confirm

### **Step 6: Configure the Web App**

You'll now be on the configuration page. Set these values:

#### **Source code:**
```
/home/hscmatch/12SE_HSC_subject_selection
```

#### **Working directory:**
```
/home/hscmatch/12SE_HSC_subject_selection
```

#### **Virtualenv:**
```
/home/hscmatch/.virtualenvs/hscmatch-venv
```

### **Step 7: Edit the WSGI File**

1. On the Web tab, find the **"Code"** section
2. Click the WSGI configuration file link (looks like `/var/www/hscmatch_pythonanywhere_com_wsgi.py`)
3. **Delete everything** in that file
4. Replace with this:

```python
import sys

# Add project directory to Python path
project_home = '/home/hscmatch/12SE_HSC_subject_selection'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Import your Flask app
from app import app as application
```

5. Click **"Save"** (top right)

### **Step 7b: Set Up config.py (API Keys and Secrets)** 🔐

The app uses Google Gemini for AI explanations and needs configuration for admin access. You need to create a `config.py` file (which is intentionally **gitignored** so it never reaches GitHub).

#### **Get a free Gemini API key**
1. Visit https://aistudio.google.com/app/apikey
2. Click **"Create API key"**
3. Copy the key (starts with `AI...`)

#### **Create `config.py` on PythonAnywhere**

In the Bash console:

```bash
cd ~/12SE_HSC_subject_selection
nano config.py
```

Paste this (replace with your actual values):

```python
GEMINI_API_KEY  = "AI...your-real-key-here..."
ADMIN_PASSWORD  = "your-admin-password"
SECRET_KEY      = "your-secret-key"
```

Save and exit: `Ctrl+O`, `Enter`, `Ctrl+X`.

> **🛡️ Graceful degradation:** If you skip this step, the app still works perfectly — AI explanations will use a fallback template and the admin password will default to "admin". You can add the keys later anytime.

### **Step 8: Configure Static Files**

This makes CSS and other static files load fast.

On the Web tab, scroll to **"Static files"** section and add:

| URL | Directory |
|-----|-----------|
| `/static/` | `/home/hscmatch/12SE_HSC_subject_selection/static/` |

Click **"Add a new static file mapping"** if there's no row, then enter the values.

### **Step 9: Reload Your Web App**

Scroll to the top of the Web tab and click the big green **"Reload hscmatch.pythonanywhere.com"** button.

### **Step 10: Visit Your Site!** 🎉

Open: https://hscmatch.pythonanywhere.com/

You should see your HSC Subject Recommendation Engine running live!

---

## 🔄 Updating After Code Changes (Do This Each Update)

Once code is pushed to GitHub, PythonAnywhere needs to **pull** the changes and **reload** the web app. This is a 30-second process.

### ⚡ Quick Update Workflow (3 Steps)

#### **Step 1: On your local machine**

Make changes → commit → push (you already do this):

```powershell
git add .
git commit -m "Your changes"
git push
```

#### **Step 2: On PythonAnywhere Bash console**

```bash
cd ~/12SE_HSC_subject_selection
git pull
```

#### **Step 3: Reload the web app**

Two ways to reload:

**Option A: From the Web tab** (easier)
- Go to https://www.pythonanywhere.com/user/hscmatch/webapps/
- Click the green **"Reload hscmatch.pythonanywhere.com"** button

**Option B: From Bash** (faster)
```bash
touch /var/www/hscmatch_pythonanywhere_com_wsgi.py
```
This "touches" the WSGI file, triggering an auto-reload.

✅ That's it! Your changes are now live at https://hscmatch.pythonanywhere.com/

---

### 🎯 Special Cases

#### **If you changed `requirements.txt`** (added/upgraded packages)

Add a `pip install` step:

```bash
workon hscmatch-venv
cd ~/12SE_HSC_subject_selection
git pull
pip install -r requirements.txt
```

Then reload.

#### **If you retrained models locally** (`models.pkl` changed)

**Option 1: Pull the new pickle (easy)**
```bash
cd ~/12SE_HSC_subject_selection
git pull
```
Then reload. Done.

**Option 2: Retrain on PythonAnywhere** (cleaner)
- Visit https://hscmatch.pythonanywhere.com/admin
- Click **"🔄 Retrain Models"** button
- Done — no Bash needed

#### **If changes don't appear after reload**
- Clear your browser cache (Ctrl+Shift+R)
- Check the error logs (Web tab → "Log files")

#### **If git pull fails with "local changes would be overwritten"**
This happens if you have local changes (e.g., from model training on PythonAnywhere). Two options:

**Option A: Stash and pull (keeps changes for later)**
```bash
git stash
git pull
```
Then reload the web app. The stashed changes are saved if needed later.

**Option B: Discard GitHub version (for training_data.csv)**
If you want to keep the PythonAnywhere version (e.g., `training_data.csv` with accumulated student data) and discard the older GitHub version:
```bash
git checkout -- training_data.csv
git pull
```
This keeps your local data and pulls other code changes.

---

### 🚀 One-Liner for Future Updates

Your magic deployment command — run this on PythonAnywhere Bash after every `git push`:

```bash
cd ~/12SE_HSC_subject_selection && git pull && touch /var/www/hscmatch_pythonanywhere_com_wsgi.py
```

#### **What each part does**

| Part | What It Does |
|------|--------------|
| `cd ~/12SE_HSC_subject_selection` | Navigate to your project folder |
| `&&` | Only run next command if previous succeeded |
| `git pull` | Download latest commits from GitHub |
| `&&` | Only reload if pull succeeded (smart!) |
| `touch /var/www/...wsgi.py` | Trigger PythonAnywhere to reload your app |

The `&&` chaining is clever — if `git pull` fails (e.g. merge conflict), it **won't** reload a broken state.

#### **Exception: when `requirements.txt` changes**

If you added/removed packages locally, add a `pip install` step:

```bash
cd ~/12SE_HSC_subject_selection && git pull && pip install --user -r requirements.txt && touch /var/www/hscmatch_pythonanywhere_com_wsgi.py
```

#### **💡 Pro Tip: Make It a Bash Alias**

Tired of typing the long command? Set up a `deploy` alias once:

```bash
echo "alias deploy='cd ~/12SE_HSC_subject_selection && git pull && touch /var/www/hscmatch_pythonanywhere_com_wsgi.py'" >> ~/.bashrc
source ~/.bashrc
```

Then in any future bash console, just type:

```bash
deploy
```

Done. ⚡ Saves seconds every deploy.

---

### 📝 Update Action Cheat Sheet

| Scenario | Command |
|----------|---------|
| **Code/HTML/CSS change** | `cd ~/12SE_HSC_subject_selection && git pull && touch /var/www/hscmatch_pythonanywhere_com_wsgi.py` |
| **Added a new package** | Same as above + `pip install --user -r requirements.txt` between pull and touch |
| **Changed `training_data.csv`** | Run deploy command, then click "🔄 Retrain Models" on `/admin` |
| **Updated model files locally** | Run deploy command (models are in repo) |
| **Changed `config.py` (API keys only)** | Just `touch /var/www/hscmatch_pythonanywhere_com_wsgi.py` — no `git pull` needed (config.py is local-only) |

---

## 🛠️ Troubleshooting

### **Site shows "Something went wrong" / 500 error**

1. Go to **Web** tab
2. Scroll to **"Log files"** section
3. Click the **error log** link
4. Read the error at the bottom (most recent entry)

Common fixes:
- **ModuleNotFoundError:** Run `pip install -r requirements.txt` in the virtualenv
- **TemplateNotFound:** Check that source code path is correct
- **InconsistentVersionWarning (sklearn):** Retrain models via the admin dashboard at `/admin`

### **CSS/styling not loading**

- Check Static Files mapping is set: `/static/` → `/home/hscmatch/12SE_HSC_subject_selection/static/`
- Reload web app

### **Changes not appearing**

- Did you click the **Reload** button on the Web tab? PythonAnywhere caches everything until you reload.

### **Charts not rendering on admin page**

- Charts use matplotlib — should work on PythonAnywhere by default
- If issue persists, retrain models: click "🔄 Retrain Models" button on the admin dashboard

### **`pip install` stuck on "Preparing metadata (pyproject.toml)..."**

This means pip is trying to **compile pandas/numpy from source** — very slow on free tier (10-20 mins) and may hang indefinitely.

**Fix:** Cancel with `Ctrl+C` and recreate the venv with `--system-site-packages`:

```bash
deactivate
rmvirtualenv hscmatch-venv   # or: rm -rf ~/.virtualenvs/hscmatch-venv (faster)
mkvirtualenv --python=/usr/bin/python3.13 --system-site-packages hscmatch-venv
cd ~/12SE_HSC_subject_selection
pip install Flask==3.0.3
```

This inherits PythonAnywhere's pre-built numpy/pandas/matplotlib/scikit-learn — instant setup.

### **Need to retrain models on PythonAnywhere**

Use the admin dashboard's **"🔄 Retrain Models"** button, or run in Bash:

```bash
workon hscmatch-venv
cd ~/12SE_HSC_subject_selection
# Retrain via admin dashboard at /admin instead
```

Then reload the web app.

---

## 📦 Environment Versions (Already Aligned)

Your local and PythonAnywhere environments match:

| Package | Version |
|---------|---------|
| Python | 3.13 |
| Flask | 3.0.3 |
| scikit-learn | 1.6.0 |
| pandas | 2.2.2 |
| numpy | 2.1.0 |
| matplotlib | 3.9.2 |

All pinned in `requirements.txt`.

---

## 🔒 Security Notes

- The admin dashboard at `/admin` is currently **publicly accessible** with no authentication
- For a production deployment, consider adding password protection
- The CSV training data (`data.csv`) is included in the repo — make sure no real student data is in it!

---

## 🎓 Free vs. Paid PythonAnywhere

**Free tier limits:**
- 1 web app
- 512 MB disk space (plenty for this project)
- Custom domain not supported (must use `trueloveai.pythonanywhere.com`)
- App goes to sleep after 3 months of inactivity (just log in to keep it active)

**Paid tier perks (if needed):**
- Always-on tasks
- Custom domains
- More CPU/storage

For this educational project, **free tier is perfect**.

---

## 📚 Useful PythonAnywhere Commands

```bash
# List your virtualenvs
lsvirtualenv

# Switch to a virtualenv
workon hscmatch-venv

# Deactivate virtualenv
deactivate

# Check Python version
python --version

# View installed packages
pip list

# Disk usage
du -sh ~/12SE_HSC_subject_selection
```

---

## 🎯 Quick Reference Card

**Your URLs:**
- Live site: https://hscmatch.pythonanywhere.com/
- Admin dashboard: https://hscmatch.pythonanywhere.com/admin
- Dashboard: https://www.pythonanywhere.com/user/hscmatch/

**Update workflow:**
1. Edit code locally → commit → push to GitHub
2. PythonAnywhere Bash: `cd ~/12SE_HSC_subject_selection && git pull`
3. Web tab → Reload button

That's the entire deployment loop. 🎉
