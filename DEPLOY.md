# GitHub + Streamlit Cloud deployment (step-by-step)

Repository name (recommended): **`ptc-recurrence-top6-streamlit`**
GitHub account: **`zsgk2003`**

Local folder is already a git repo with one commit on branch `main`.

---

## Step 1: Create empty GitHub repository

1. Open https://github.com/new
2. Owner: `zsgk2003`
3. Repository name: `ptc-recurrence-top6-streamlit`
4. Visibility: **Public** (required for free Streamlit Community Cloud)
5. Do **not** add README, .gitignore, or license (repo must stay empty)
6. Click **Create repository**

---

## Step 2: Push from this PC

Double-click **`push_to_github.bat`**, or run in PowerShell:

```powershell
cd "D:\??_??\windows????20260516\01_thyroid cancer recurrence\02_PTC_top6_streamlit_app"
git remote remove origin
git remote add origin https://github.com/zsgk2003/ptc-recurrence-top6-streamlit.git
git push -u origin main
```

If Git asks for credentials, use a **Personal Access Token** (not your password):
https://github.com/settings/tokens -> Generate new token (classic) -> scope `repo`

---

## Step 3: Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub
2. **Create app**
3. Repository: `zsgk2003/ptc-recurrence-top6-streamlit`
4. Branch: `main`
5. **Main file path**: `app.py`
6. (Optional) App URL slug: e.g. `ptc-recurrence-top6`
7. Click **Deploy**

Streamlit installs packages from `requirements.txt`. No secrets needed.

**Advanced settings** (only if build fails): set Python version to **3.11**.

---

## Step 4: Verify live app

- Open the public URL (e.g. `https://ptc-recurrence-top6.streamlit.app`)
- Sidebar should show test AUC **0.9905**
- Single Prediction with defaults -> low risk (~0%)
- Model Performance -> ROC / SHAP charts load

---

## What is in the repo

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `train_model.py` | Reproduce model; `fit_pipeline()` for cloud fallback |
| `artifacts/model_LightGBM_top6.pkl` | Pre-trained model (fast cold start) |
| `data/modeldata_335_PTC.csv` | Training data if pickle cannot load |
| `requirements.txt` | Cloud dependencies |
| `.streamlit/config.toml` | Theme and server defaults |

If the pickle fails on cloud (library mismatch), the app **retrains automatically** on first visit (may take 1-2 minutes).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `git push` connection reset | VPN/proxy; retry; or push from another network |
| 404 on push | Create the empty repo on GitHub first |
| Streamlit build fails on LightGBM | Set Python 3.11 in app settings |
| Slow first load | Normal if retraining; commit includes `.pkl` to avoid this |
