## Setting Up Virtual Environment

Create a virtual environment : 
```bash
python -m venv <VIRTUAL_ENVIRONMENT_NAME>
```
Activate the virtual environment :
```bash 
<VIRTUAL_ENVIRONMENT_NAME>\Scripts\activate
```

## Package Installation
Upgrade pip to the latest version : 
```bash
pip install --upgrade pip
```

Install required packages :
```bash
pip install -e .[extras]  
pip install psycopg2 httpx
```

## Running MLflow Server
### Basic Server Start
Start the MLflow server with Keycloak authentication:
```bash
mlflow server --app-name keycloak-auth
```
### Server with Database Backend
If you want to start the backend with a database, use the following command :
```bash
mlflow server --backend-store-uri <dialect>+<driver>://<username>:<password>@<host>:<port>/<database> --default-artifact-root S3://bucketname --app-name keycloak-auth
```
Replace the placeholders with your specific database connection details:

- `<dialect>` : The type of database (e.g., postgresql).
- `<driver>` : The database driver (optional).
- `<username>` : Your database username.
- `<password>` : Your database password.
- `<host>` : The database host.
- `<port>` : The database port.
- `<database>` : The database name.

>**Note** :- Activate the virtual environment (if not already activated) : `<VIRTUAL_ENVIRONMENT_NAME>\Scripts\activate`
