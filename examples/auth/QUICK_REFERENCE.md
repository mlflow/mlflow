# MLflow Azure SSO - Quick Reference Card

## 🎯 3-Step Setup

### Step 1: Azure Portal (Copy these values)
```
Tenant ID (Directory ID):    ________________________
Client ID (App ID):          ________________________
Client Secret:               ________________________
```

### Step 2: Environment Setup
```bash
cd examples/auth
cp .env.azure_sso.example .env.azure_sso
# Edit with your values
```

### Step 3: Start MLflow
```bash
docker-compose -f docker-compose.azure_sso.yml up -d
```

Then visit: `http://localhost:5000`

---

## 📋 Configuration Checklist

| Item | Value | Done |
|------|-------|------|
| Azure Tenant ID | `AZURE_TENANT_ID` | [ ] |
| Azure Client ID | `AZURE_CLIENT_ID` | [ ] |
| Azure Client Secret | `AZURE_CLIENT_SECRET` | [ ] |
| Redirect URI | `http://localhost:5000/login/callback` | [ ] |
| API Permission | `User.Read` (Microsoft Graph) | [ ] |
| MLflow Config | `auth_azure.ini` copied | [ ] |
| Environment | `.env.azure_sso` created | [ ] |
| Docker | Running and healthy | [ ] |

---

## 🔧 Common Commands

### Docker Operations
```bash
# Start
docker-compose -f docker-compose.azure_sso.yml up -d

# View logs
docker-compose -f docker-compose.azure_sso.yml logs -f mlflow

# Stop
docker-compose -f docker-compose.azure_sso.yml down

# Reset data
docker-compose -f docker-compose.azure_sso.yml down -v
```

### Local Python Setup
```bash
# Install
pip install -r requirements_azure_sso.txt

# Set environment
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
export MLFLOW_AUTH_CONFIG_PATH="$(pwd)/auth_azure.ini"

# Start
mlflow server --app-name basic-auth --host 0.0.0.0 --port 5000
```

### Test Connection
```python
import mlflow
import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "user@tenant.onmicrosoft.com"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "access-token"
mlflow.set_tracking_uri("http://localhost:5000")

experiment = mlflow.set_experiment("test")
print(f"✓ Connected: {experiment.name}")
```

---

## 🐛 Troubleshooting Matrix

| Issue | Solution |
|-------|----------|
| **Unauthorized** | Check credentials in Azure Portal |
| **Port 5000 in use** | `lsof -ti:5000 \| xargs kill -9` |
| **Module not found** | `pip install -r requirements_azure_sso.txt` |
| **Invalid tenant** | Verify `AZURE_TENANT_ID` in Azure Portal |
| **Container won't start** | Check logs: `docker-compose logs mlflow` |
| **Database locked** | Stop containers: `docker-compose down` |
| **Token expired** | Refresh token from Azure |

---

## 📚 Documentation Map

```
START HERE
    ↓
AZURE_SSO_QUICKSTART.md (15 min)
    ↓
    ├─→ Works? ✓ Go to examples: azure_sso_example.py
    │
    └─→ Need more? → AZURE_SSO_SETUP.md (full production guide)
         │
         ├─→ Questions? → README_AZURE_SSO.md (detailed reference)
         │
         └─→ Lost? → AZURE_SSO_SETUP_SUMMARY.md (overview)
```

---

## 🔐 Security Checklist

Before production deployment:

- [ ] Change admin password in `auth_azure.ini`
- [ ] Use HTTPS with valid certificates
- [ ] Store secrets in secure vault (not in code)
- [ ] Enable database backups
- [ ] Configure firewall rules
- [ ] Set up monitoring and alerts
- [ ] Test with multiple users
- [ ] Review all permissions
- [ ] Implement RBAC
- [ ] Document access procedures

---

## 📞 File Quick Links

| Need | File |
|------|------|
| Fast setup | `AZURE_SSO_QUICKSTART.md` |
| Full guide | `../AZURE_SSO_SETUP.md` |
| Overview | `../AZURE_SSO_SETUP_SUMMARY.md` |
| Examples | `azure_sso_example.py` |
| Config | `auth_azure.ini` |
| Docker | `docker-compose.azure_sso.yml` |
| Env vars | `.env.azure_sso.example` |
| Directory | `README_AZURE_SSO.md` |

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Read this card | 2 min |
| Azure Portal setup | 10 min |
| Start Docker | 5 min |
| Test connection | 5 min |
| Review full guide | 30 min |
| **Total** | **52 min** |

---

## ✅ Verification Steps

After setup, verify each step:

```bash
# 1. Check Docker is running
docker-compose ps

# 2. Check MLflow is responding
curl http://localhost:5000

# 3. Check authentication module
ls -l mlflow/server/auth/azure_auth.py

# 4. Check configuration
cat auth_azure.ini

# 5. Check environment
cat .env.azure_sso
```

---

## 🎓 Learning Order

1. This card (2 min)
2. `AZURE_SSO_QUICKSTART.md` (15 min)
3. Try local setup (10 min)
4. Run examples (5 min)
5. Read `AZURE_SSO_SETUP.md` (30 min)
6. Deploy to production

---

## 💡 Pro Tips

- **Use .env file**: `source .env.azure_sso` to load all vars at once
- **Enable debug logging**: Set `MLFLOW_LOGGING_LEVEL=DEBUG`
- **Token caching**: Set `auth_cache_ttl_seconds = 3600` for production
- **Database backups**: Regular PostgreSQL backups recommended
- **Multiple users**: Test with different Azure AD users before production
- **Monitor logs**: Watch logs while testing: `docker-compose logs -f`

---

## 🚀 You're Ready!

Everything is set up. Start with:

```bash
cd examples/auth
docker-compose -f docker-compose.azure_sso.yml up -d
echo "✓ MLflow running at http://localhost:5000"
```

Have questions? Check `README_AZURE_SSO.md`

---

**Quick Reference v1.0** | Last Updated: September 25, 2026
