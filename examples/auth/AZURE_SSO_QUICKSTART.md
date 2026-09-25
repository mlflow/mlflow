# MLflow Azure SSO - Quick Start Guide

This guide will help you set up MLflow with Azure AD (Entra ID) SSO authentication in **under 15 minutes**.

## What You'll Need

- Azure Portal access (global admin or app admin role)
- Docker and Docker Compose (or Python 3.10+)
- Terminal/Command Prompt

## Step 1: Register Application in Azure Portal (3 minutes)

### 1.1 Create App Registration

```
Azure Portal → Azure Active Directory → App registrations → New registration
```

Fill in:
- **Name**: `MLflow`
- **Supported account types**: `Accounts in this organizational directory only`
- Click **Register**

### 1.2 Copy Credentials

After registration, you'll see the Overview page. Copy these values:

```
Application (client) ID:    ____________________
Directory (tenant) ID:       ____________________
```

### 1.3 Create Client Secret

1. Go to **Certificates & secrets**
2. Click **New client secret**
3. Description: `MLflow Authentication`
4. Expires: `24 months` (or your preference)
5. Click **Add**
6. **Copy the secret value immediately** (it won't show again!)

```
Client secret value:          ____________________
```

### 1.4 Configure Redirect URI

1. Go to **Authentication**
2. Click **Add a platform**
3. Select **Web**
4. Redirect URI: `http://localhost:5000/login/callback`
5. Click **Configure**

### 1.5 Grant API Permissions

1. Go to **API permissions**
2. Click **Add a permission**
3. Select **Microsoft Graph**
4. Select **Delegated permissions**
5. Search and select `User.Read`
6. Click **Add permissions**

## Step 2: Set Up MLflow Locally (5 minutes)

### Option A: Using Docker Compose (Recommended)

```bash
cd examples/auth

# Copy and edit the environment file
cp .env.azure_sso.example .env.azure_sso

# Edit .env.azure_sso with your Azure credentials
# AZURE_TENANT_ID=your-tenant-id
# AZURE_CLIENT_ID=your-client-id
# AZURE_CLIENT_SECRET=your-client-secret
# AZURE_REDIRECT_URI=http://localhost:5000/login/callback

# Load environment
export $(cat .env.azure_sso | xargs)

# Start MLflow
docker-compose -f docker-compose.azure_sso.yml up -d

# Check status
docker-compose -f docker-compose.azure_sso.yml ps

# View logs
docker-compose -f docker-compose.azure_sso.yml logs -f mlflow
```

### Option B: Using Python (Local)

```bash
cd examples/auth

# Install dependencies
pip install mlflow msal requests python-jose cryptography

# Set environment variables
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
export MLFLOW_AUTH_CONFIG_PATH="$(pwd)/auth_azure.ini"

# Start MLflow
mlflow server --app-name basic-auth --host 0.0.0.0 --port 5000
```

## Step 3: Access MLflow (2 minutes)

1. Open browser: `http://localhost:5000`
2. You should see the MLflow UI
3. Go to **Models** section to verify authentication is working

## Step 4: Test Authentication (2 minutes)

### Test with Python Script

```python
import os
import mlflow

# Set credentials (use an Azure user account)
os.environ["MLFLOW_TRACKING_USERNAME"] = "your-email@tenant.onmicrosoft.com"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "your-azure-password-or-token"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# Test connection
mlflow.set_tracking_uri("http://localhost:5000")
experiment = mlflow.set_experiment("test_azure_sso")

print(f"✓ Successfully authenticated!")
print(f"✓ Experiment created: {experiment.name}")
```

### Or Use the Example Script

```bash
# Run the provided example
python azure_sso_example.py
```

## Troubleshooting

### Error: "Unauthorized"

**Problem**: Authentication is failing
- Check that AZURE_CLIENT_ID and AZURE_CLIENT_SECRET are correct
- Verify the user exists in your Azure AD directory
- Check that the redirect URI matches what's configured in Azure Portal

### Error: "Module not found"

**Problem**: Missing Python dependencies
```bash
pip install msal requests python-jose cryptography
```

### Error: "Invalid tenant"

**Problem**: AZURE_TENANT_ID is incorrect
- Go to Azure Portal → Properties
- Copy the exact **Tenant ID** value

### MLflow Shows No Authentication

**Problem**: Authorization function not loaded
- Verify `MLFLOW_AUTH_CONFIG_PATH` points to correct `auth_azure.ini`
- Check that `mlflow/server/auth/azure_auth.py` is installed
- Look at MLflow logs for errors

### Port Already in Use

**Problem**: Another service using port 5000
```bash
# Use different port
mlflow server --app-name basic-auth --port 5001

# Or kill existing process
lsof -ti:5000 | xargs kill -9
```

## Next Steps

1. **Add More Users**: Go to Azure Portal → Users and add team members
2. **Configure Permissions**: Use MLflow UI to grant permissions to users
3. **Production Deployment**: 
   - Use HTTPS with valid certificates
   - Use PostgreSQL or cloud database instead of SQLite
   - Set strong admin password
   - Configure firewall rules
4. **Enable Workspaces**: Set `MLFLOW_ENABLE_WORKSPACES=true` for multi-tenant setup

## Common Azure User Formats

Use these formats when setting `MLFLOW_TRACKING_USERNAME`:

| Format | Example | When to Use |
|--------|---------|-------------|
| User Principal Name | `user@tenant.onmicrosoft.com` | Most common |
| Email | `user@company.com` | If custom domain configured |
| Object ID | `12345678-1234-1234-1234-123456789012` | Programmatic access |

## Testing with Bearer Token

For API calls, use the access token:

```bash
# Get token (using Azure CLI)
az account get-access-token --resource https://graph.microsoft.com --query accessToken -o tsv

# Use in requests
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:5000/api/2.0/mlflow/experiments/list
```

## Cleanup

```bash
# Stop containers
docker-compose -f docker-compose.azure_sso.yml down

# Remove volumes (careful: deletes data)
docker-compose -f docker-compose.azure_sso.yml down -v

# Or stop local server
# Press Ctrl+C to stop MLflow server
```

## Security Best Practices

1. **Never commit secrets**: Use `.env` files, not code
2. **Use strong passwords**: Admin password should be 16+ characters
3. **Enable HTTPS**: Use certificates in production
4. **Rotate secrets**: Change client secret regularly
5. **Monitor access**: Check MLflow logs for unauthorized attempts
6. **Principle of least privilege**: Grant minimum required permissions
7. **Enable audit logging**: Track who accesses what

## Performance Tips

1. **Enable caching**: Set `auth_cache_ttl_seconds = 3600` in config
2. **Use read replicas**: Set `read_database_uri` for analytics queries
3. **Scale horizontally**: Run multiple MLflow instances behind load balancer
4. **Monitor metrics**: Track authentication response times

## Getting Help

- Check MLflow logs: `docker-compose logs mlflow` or server console
- Review Azure Portal > App registrations > MLflow > Overview
- Check Azure AD → Users for account details
- Enable debug logging: `MLFLOW_LOGGING_LEVEL=DEBUG`

## Next: Production Deployment

See [AZURE_SSO_SETUP.md](../../AZURE_SSO_SETUP.md) for production deployment with:
- HTTPS/SSL certificates
- PostgreSQL database
- Load balancing
- Monitoring and alerts
- RBAC (Role-Based Access Control)
