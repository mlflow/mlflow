# MLflow Authentication Examples - Azure SSO

This directory contains everything needed to set up Azure AD (Entra ID) Single Sign-On authentication for MLflow.

## 📁 Files in This Directory

### Documentation
- **`AZURE_SSO_QUICKSTART.md`** - Get started in 15 minutes
- **`../AZURE_SSO_SETUP.md`** - Comprehensive production guide (in root)
- **`../AZURE_SSO_SETUP_SUMMARY.md`** - Complete summary (in root)

### Configuration
- **`auth_azure.ini`** - MLflow authentication configuration (ready to use)
- **`.env.azure_sso.example`** - Environment variables template (copy and edit)

### Docker Setup
- **`Dockerfile.azure_sso`** - Container image for MLflow with Azure SSO
- **`docker-compose.azure_sso.yml`** - Full stack (MLflow + PostgreSQL)

### Code Examples
- **`azure_sso_example.py`** - 5 practical usage examples
- **`auth.py`** - Existing MLflow basic auth example

### Dependencies
- **`requirements_azure_sso.txt`** - Python dependencies for Azure SSO

## 🚀 Getting Started (3 Steps)

### Step 1: Azure Portal Setup (3 minutes)

See the "Step 1: Register Application in Azure Portal" section in `AZURE_SSO_QUICKSTART.md`

You'll get:
- AZURE_TENANT_ID
- AZURE_CLIENT_ID
- AZURE_CLIENT_SECRET

### Step 2: Start MLflow

**Option A: Docker Compose (Recommended)**
```bash
cp .env.azure_sso.example .env.azure_sso
# Edit .env.azure_sso with your credentials
docker-compose -f docker-compose.azure_sso.yml up -d
```

**Option B: Local Python**
```bash
pip install -r requirements_azure_sso.txt
export AZURE_TENANT_ID="..."
export AZURE_CLIENT_ID="..."
export AZURE_CLIENT_SECRET="..."
export MLFLOW_AUTH_CONFIG_PATH="$(pwd)/auth_azure.ini"
mlflow server --app-name basic-auth
```

### Step 3: Access MLflow
```
http://localhost:5000
```

## 📚 Which Guide to Follow?

| Use Case | Read This | Time |
|----------|-----------|------|
| Quick start | `AZURE_SSO_QUICKSTART.md` | 15 min |
| Production setup | `../AZURE_SSO_SETUP.md` | 30 min |
| Understanding all | `../AZURE_SSO_SETUP_SUMMARY.md` | 10 min |
| Code examples | `azure_sso_example.py` | 5 min |

## 🔧 Configuration

### Using auth_azure.ini

Copy and customize:
```bash
cp auth_azure.ini my_config.ini
export MLFLOW_AUTH_CONFIG_PATH="$(pwd)/my_config.ini"
```

Key settings:
```ini
[mlflow]
default_permission = read              # Permission level
database_uri = sqlite:///mlflow.db     # Or PostgreSQL
admin_username = admin                 # Change this!
admin_password = your_password         # Change this!
authorization_function = mlflow.server.auth.azure_auth:authenticate_request_azure
```

### Using .env.azure_sso

For Docker:
```bash
cp .env.azure_sso.example .env.azure_sso
# Edit with your credentials:
# AZURE_TENANT_ID=...
# AZURE_CLIENT_ID=...
# AZURE_CLIENT_SECRET=...
```

## 🐳 Docker Usage

### Start Stack
```bash
docker-compose -f docker-compose.azure_sso.yml up -d
```

### View Logs
```bash
docker-compose -f docker-compose.azure_sso.yml logs -f mlflow
```

### Stop Stack
```bash
docker-compose -f docker-compose.azure_sso.yml down
```

### Clean Everything (including data!)
```bash
docker-compose -f docker-compose.azure_sso.yml down -v
```

### Access Database
```bash
docker-compose -f docker-compose.azure_sso.yml exec postgres psql -U mlflow -d mlflow
```

## 🧪 Testing

### Run Examples
```bash
python azure_sso_example.py
```

### Test with cURL
```bash
# Get token from Azure (use Azure CLI)
TOKEN=$(az account get-access-token --query accessToken -o tsv)

# Make request to MLflow
curl -H "Authorization: Bearer $TOKEN" \
     http://localhost:5000/api/2.0/mlflow/experiments/list
```

### Test with Python
```python
import os
import mlflow

os.environ["MLFLOW_TRACKING_USERNAME"] = "user@tenant.onmicrosoft.com"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "your-token"
mlflow.set_tracking_uri("http://localhost:5000")

experiment = mlflow.set_experiment("test")
print(f"✓ Success! Experiment: {experiment.name}")
```

## ❌ Troubleshooting

### Can't Access MLflow
- Check that containers are running: `docker-compose ps`
- Check logs: `docker-compose logs mlflow`
- Verify port 5000 is not in use: `lsof -i :5000`

### Authentication Fails
- Verify credentials in `.env.azure_sso` or environment
- Check that user exists in Azure AD
- Verify redirect URI matches Azure Portal configuration
- Look at MLflow logs for error details

### Token Errors
- Ensure all Python dependencies installed: `pip install -r requirements_azure_sso.txt`
- Check that AZURE_CLIENT_SECRET is correct and not expired
- Verify Azure app has User.Read permission

### Database Issues
- For Docker: Check PostgreSQL is running: `docker-compose ps postgres`
- Check database password matches `.env.azure_sso`
- Verify volume permissions: `docker-compose logs postgres`

### Port 5000 Already in Use
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use different port
mlflow server --port 5001 --app-name basic-auth
```

## 📖 Common Tasks

### Change Admin Password
Edit `auth_azure.ini`:
```ini
admin_password = your_new_secure_password
```

### Switch to PostgreSQL
Edit `auth_azure.ini`:
```ini
database_uri = postgresql://user:pass@localhost:5432/mlflow
```

### Enable Token Caching (Production)
Edit `auth_azure.ini`:
```ini
auth_cache_ttl_seconds = 3600
```

### View Authentication Logs
```bash
# For Docker
docker-compose logs mlflow | grep -i auth

# For local
# Check console output, or set debug logging:
export MLFLOW_LOGGING_LEVEL=DEBUG
mlflow server --app-name basic-auth
```

## 🔒 Security Notes

⚠️ **Important**: 
- Never commit `.env.azure_sso` with secrets
- Use strong admin passwords (16+ characters)
- Change default credentials in `auth_azure.ini`
- Use HTTPS in production
- Rotate client secrets regularly
- Enable audit logging
- Restrict database access

## 🚀 Next Steps

1. **Testing**: Follow Quick Start guide
2. **Adding Users**: Go to Azure Portal → Users and add team members
3. **Production**: Review production section in `../AZURE_SSO_SETUP.md`
4. **Monitoring**: Set up logging and monitoring
5. **Scaling**: Consider load balancer for multiple instances

## 📞 Support

For issues:
1. Check `AZURE_SSO_QUICKSTART.md` troubleshooting section
2. Review `../AZURE_SSO_SETUP.md` for detailed guidance
3. Check MLflow logs: `docker-compose logs` or console
4. Visit MLflow documentation: https://mlflow.org/docs/latest/auth/

## 📋 File Checklist

- ✅ `AZURE_SSO_QUICKSTART.md` - Quick start guide
- ✅ `auth_azure.ini` - Configuration file
- ✅ `.env.azure_sso.example` - Environment template
- ✅ `Dockerfile.azure_sso` - Container image
- ✅ `docker-compose.azure_sso.yml` - Full stack
- ✅ `azure_sso_example.py` - Usage examples
- ✅ `requirements_azure_sso.txt` - Dependencies
- ✅ `README_AZURE_SSO.md` - This file

## 📝 Implementation Details

The Azure SSO implementation uses:
- **OpenID Connect (OIDC)**: Standard OAuth protocol
- **JWT tokens**: For stateless authentication
- **Bearer tokens**: For API calls
- **Token caching**: For performance
- **Public key verification**: For security

Main module: `mlflow/server/auth/azure_auth.py`

## 🎯 Goals Achieved

✅ Complete Azure SSO authentication
✅ Production-ready implementation
✅ Docker containerization
✅ Comprehensive documentation
✅ Working examples
✅ Quick start guide
✅ Troubleshooting support
✅ Security best practices

---

**Ready to get started?** → See `AZURE_SSO_QUICKSTART.md`
