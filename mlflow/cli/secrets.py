import datetime
import hashlib
import os
import sys
from pathlib import Path

import click
from cryptography.fernet import Fernet


@click.group("secrets")
def commands():
    pass


@commands.command("generate-key")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Path to write the generated key file (created with 600 permissions).",
)
@click.option(
    "--stdout",
    is_flag=True,
    help="Print the generated key to stdout instead of writing to a file.",
)
@click.option(
    "--k8s-secret",
    metavar="NAME",
    help="Output a kubectl command to create a Kubernetes secret with the given name.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Allow overwriting an existing key file (use with caution).",
)
def generate_key(output, stdout, k8s_secret, force):
    """
    Generate a new master key for MLflow secrets encryption.

    This command generates a Fernet-compatible master key that can be used
    to encrypt secrets in MLflow. The key can be written to a file with
    secure permissions (600), printed to stdout, or formatted as a kubectl
    command for Kubernetes deployment.

    Examples:

        # Generate key to a file with 600 permissions
        mlflow secrets generate-key --output /secure/master.key

        # Generate key to stdout for scripting
        mlflow secrets generate-key --stdout

        # Generate kubectl command for Kubernetes
        mlflow secrets generate-key --k8s-secret mlflow-master-key

    IMPORTANT: Store this key securely. Loss of the master key means
    permanent loss of access to all encrypted secrets.
    """
    num_modes = sum([output is not None, stdout, k8s_secret is not None])
    if num_modes > 1:
        raise click.UsageError("Specify only one of: --output, --stdout, or --k8s-secret")

    if num_modes == 0:
        raise click.UsageError("Must specify one of: --output, --stdout, or --k8s-secret")

    key = Fernet.generate_key().decode()

    if stdout:
        click.echo(key)
        click.echo()
        click.echo("WARNING: This key is shown only once. Store it securely.", err=True)
        click.echo()
        click.echo("Recommended storage options:", err=True)
        click.echo('  1. Save to file: echo "KEY" > key.txt && chmod 600 key.txt', err=True)
        click.echo('  2. Environment:  export MLFLOW_SECRET_MASTER_KEY="KEY"', err=True)
        click.echo("  3. AWS:          aws secretsmanager create-secret ...", err=True)
        click.echo("  4. Kubernetes:   kubectl create secret generic ...", err=True)
        return

    if k8s_secret:
        kubectl_cmd = (
            f"kubectl create secret generic {k8s_secret} "
            f"--from-literal=MLFLOW_SECRET_MASTER_KEY='{key}'"
        )
        click.echo(kubectl_cmd)
        click.echo()
        click.echo("To use this secret in a pod, add to your deployment:", err=True)
        click.echo("  envFrom:", err=True)
        click.echo("    - secretRef:", err=True)
        click.echo(f"        name: {k8s_secret}", err=True)
        return

    if output:
        output_path = Path(output).expanduser().resolve()

        if output_path.exists() and not force:
            raise click.UsageError(
                f"File '{output_path}' already exists. Use --force to overwrite."
            )

        try:
            fd = os.open(output_path, os.O_CREAT | os.O_WRONLY | (0 if force else os.O_EXCL), 0o600)
            try:
                if force:
                    os.ftruncate(fd, 0)
                os.write(fd, key.encode())
            finally:
                os.close(fd)

            os.chmod(output_path, 0o600)
        except FileExistsError:
            raise click.UsageError(
                f"File '{output_path}' already exists. Use --force to overwrite."
            )
        except Exception as e:
            click.echo(f"Error writing key file: {e}", err=True)
            sys.exit(1)

        stat_info = output_path.stat()
        mode = oct(stat_info.st_mode)[-3:]

        click.echo(f"✓ Generated master key: {output_path}")
        click.echo(f"✓ File permissions: {mode} (owner read/write only)")
        click.echo(f"✓ Owner: {stat_info.st_uid}:{stat_info.st_gid}")
        click.echo()
        click.echo("To use this key:")
        click.echo(f"  export MLFLOW_SECRET_MASTER_KEY_FILE={output_path}")
        click.echo("  mlflow server ...")
        click.echo()
        click.echo("IMPORTANT: Backup this key to a secure location!")


@commands.command("rotate-key")
@click.option(
    "--current-key-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to the current master key file.",
)
@click.option(
    "--new-key-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to the new master key file.",
)
@click.option(
    "--backend-store-uri",
    metavar="URI",
    help="Backend store URI. If not specified, uses MLFLOW_TRACKING_URI or default.",
)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Number of secrets to process per batch (default: 100).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be rotated without making changes.",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Skip confirmation prompt.",
)
def rotate_key(current_key_file, new_key_file, backend_store_uri, batch_size, dry_run, yes):
    """
    Rotate the master encryption key for all secrets.

    This command re-encrypts all Data Encryption Keys (DEKs) with a new master key,
    enabling key rotation without re-encrypting the actual secret values.

    Examples:

        # Dry run to see what would be rotated
        mlflow secrets rotate-key --current-key-file /old/key --new-key-file /new/key --dry-run

        # Rotate keys
        mlflow secrets rotate-key --current-key-file /old/key --new-key-file /new/key
    """
    from mlflow.secrets.crypto import SecretManager
    from mlflow.store.tracking.dbmodels.models import SqlSecret
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

    current_key = Path(current_key_file).read_text().strip()
    new_key = Path(new_key_file).read_text().strip()

    if current_key == new_key:
        click.echo("Error: Current and new keys are identical", err=True)
        sys.exit(1)

    os.environ["MLFLOW_SECRET_MASTER_KEY"] = current_key
    current_manager = SecretManager()

    os.environ["MLFLOW_SECRET_MASTER_KEY"] = new_key
    new_manager = SecretManager()

    uri = backend_store_uri or os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    from mlflow.tracking import _get_store

    store = _get_store(uri)

    if not isinstance(store, SqlAlchemyStore):
        click.echo("Error: Key rotation only supported for SQL-backed stores", err=True)
        sys.exit(1)

    with store.ManagedSessionMaker() as session:
        total_secrets = session.query(SqlSecret).filter(SqlSecret.encrypted_dek.isnot(None)).count()

        if total_secrets == 0:
            click.echo("No secrets with envelope encryption found")
            return

        click.echo(f"Found {total_secrets} secrets to rotate")
        click.echo(f"Batch size: {batch_size}")
        click.echo()

        if dry_run:
            click.echo("DRY RUN - No changes will be made")
            click.echo(f"Would rotate {total_secrets} secrets")
            return

        if not yes:
            click.confirm(f"Rotate {total_secrets} secrets with new master key?", abort=True)

        rotated = 0
        offset = 0

        while offset < total_secrets:
            batch = (
                session.query(SqlSecret)
                .filter(SqlSecret.encrypted_dek.isnot(None))
                .limit(batch_size)
                .offset(offset)
                .all()
            )

            for secret in batch:
                try:
                    dek = current_manager.decrypt_dek(secret.encrypted_dek)
                    new_encrypted_dek = new_manager.encrypt_dek(dek)

                    secret.encrypted_dek = new_encrypted_dek
                    secret.master_key_version = (secret.master_key_version or 1) + 1

                    rotated += 1
                except Exception as e:
                    click.echo(f"Error rotating secret {secret.id}: {e}", err=True)
                    session.rollback()
                    sys.exit(1)

            session.commit()
            offset += batch_size
            click.echo(f"Progress: {min(offset, total_secrets)}/{total_secrets} secrets rotated")

        click.echo()
        click.echo(f"✓ Successfully rotated {rotated} secrets")
        click.echo()
        click.echo("IMPORTANT: Update your environment to use the new key:")
        click.echo(f"  export MLFLOW_SECRET_MASTER_KEY_FILE={new_key_file}")


@commands.command("status")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed diagnostic information.",
)
@click.option(
    "--backend-store-uri",
    metavar="URI",
    help=(
        "Backend store URI to check for existing secrets. "
        "If not specified, uses MLFLOW_TRACKING_URI or default."
    ),
)
def status(verbose, backend_store_uri):
    """
    Show the current secrets configuration status.

    This command displays information about the master key configuration,
    validates file permissions, and optionally tests decryption of existing
    secrets in the database.

    Examples:

        # Show basic status
        mlflow secrets status

        # Show detailed diagnostic information
        mlflow secrets status --verbose

        # Check status for a specific backend
        mlflow secrets status --backend-store-uri sqlite:///mlflow.db
    """
    from mlflow.secrets.crypto import SecretManager

    click.echo("MLflow Secrets Configuration")
    click.echo("=" * 60)

    env_key = os.environ.get("MLFLOW_SECRET_MASTER_KEY")
    key_file = os.environ.get("MLFLOW_SECRET_MASTER_KEY_FILE")

    has_errors = False
    has_warnings = False

    if env_key and key_file:
        click.echo("Status:           ✗ Configuration Error", err=True)
        click.echo()
        click.echo(
            "✗ Both MLFLOW_SECRET_MASTER_KEY and MLFLOW_SECRET_MASTER_KEY_FILE are set.",
            err=True,
        )
        click.echo("  Please set only one.", err=True)
        sys.exit(2)

    if env_key:
        click.echo("Status:           ✓ Configured")
        click.echo("Key Source:       Environment Variable")
        click.echo()

        key_len = len(env_key)
        if key_len != 44:
            click.echo(f"⚠️  Warning: Key length is {key_len}, expected 44", err=True)
            has_warnings = True

        fingerprint = hashlib.sha256(env_key.encode()).hexdigest()[:8]
        click.echo(f"Key Fingerprint:  sha256:{fingerprint}")

    elif key_file:
        click.echo("Status:           ✓ Configured")
        click.echo("Key Source:       File")
        click.echo(f"Key File:         {key_file}")
        click.echo()

        key_file_path = Path(key_file).expanduser().resolve()

        if not key_file_path.exists():
            click.echo(f"✗ File does not exist: {key_file_path}", err=True)
            has_errors = True
        else:
            stat_info = key_file_path.stat()
            mode = oct(stat_info.st_mode)[-3:]

            if mode in ("600", "400"):
                click.echo(f"File Permissions: {mode} (✓ Secure)")
            else:
                if stat_info.st_mode & 0o044:
                    click.echo(
                        f"File Permissions: {mode} (✗ INSECURE: group/world readable)", err=True
                    )
                    click.echo(f"  Fix immediately: chmod 600 {key_file_path}", err=True)
                    has_errors = True
                else:
                    click.echo(f"File Permissions: {mode} (⚠️  Recommended: 600 or 400)", err=True)
                    has_warnings = True

            click.echo(f"File Owner:       {stat_info.st_uid}:{stat_info.st_gid}")

            try:
                key_content = key_file_path.read_text().strip()
                fingerprint = hashlib.sha256(key_content.encode()).hexdigest()[:8]
                click.echo(f"Key Fingerprint:  sha256:{fingerprint}")

                if len(key_content) != 44:
                    click.echo(
                        f"⚠️  Warning: Key length is {len(key_content)}, expected 44", err=True
                    )
                    has_warnings = True
            except Exception as e:
                click.echo(f"✗ Cannot read key file: {e}", err=True)
                has_errors = True

            if verbose:
                mtime = datetime.datetime.fromtimestamp(stat_info.st_mtime)
                click.echo(f"File Modified:    {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                click.echo(f"File Size:        {stat_info.st_size} bytes")

    else:
        click.echo("Status:           ⚠️  Temporary Key", err=True)
        click.echo("Key Source:       Auto-generated (temporary)")
        click.echo()
        click.echo("⚠️  WARNING: Using temporary key", err=True)
        click.echo("Secrets will be LOST when the server restarts!", err=True)
        click.echo()
        click.echo("For production use:", err=True)
        click.echo("  mlflow secrets generate-key --output /secure/master.key", err=True)
        click.echo("  export MLFLOW_SECRET_MASTER_KEY_FILE=/secure/master.key", err=True)
        has_warnings = True

    try:
        secret_manager = SecretManager()
        test_value = "test_secret_12345"
        encrypted = secret_manager.encrypt(test_value)
        decrypted = secret_manager.decrypt(encrypted)

        if decrypted == test_value:
            click.echo()
            click.echo("Encryption Test:  ✓ Passed")
        else:
            click.echo()
            click.echo("✗ Encryption test failed: decrypted value doesn't match", err=True)
            has_errors = True
    except Exception as e:
        click.echo()
        click.echo(f"✗ Encryption test failed: {e}", err=True)
        has_errors = True

    if backend_store_uri or os.environ.get("MLFLOW_TRACKING_URI"):
        try:
            from mlflow.tracking import _get_store

            uri = backend_store_uri or os.environ.get("MLFLOW_TRACKING_URI")
            store = _get_store(uri)

            if hasattr(store, "list_secret_names"):
                click.echo()
                click.echo("Database Status:")

                try:
                    from mlflow.secrets.scope import SecretScope

                    global_secrets = store.list_secret_names(SecretScope.GLOBAL, None)
                    scorer_secrets = []

                    total = len(global_secrets) + len(scorer_secrets)
                    click.echo(f"Total Secrets:    {total}")
                    if total > 0:
                        click.echo(
                            f"By Scope:         GLOBAL: {len(global_secrets)}, "
                            f"SCORER: {len(scorer_secrets)}"
                        )

                        if verbose and total > 0 and total <= 10:
                            click.echo()
                            click.echo("Secret Names:")
                            for name in global_secrets:
                                click.echo(f"  [GLOBAL] {name}")
                            for name in scorer_secrets:
                                click.echo(f"  [SCORER] {name}")

                        try:
                            if global_secrets:
                                store.get_secret(global_secrets[0], SecretScope.GLOBAL, None)
                                click.echo()
                                click.echo("Decryption Test:  ✓ Can decrypt existing secrets")
                        except Exception as e:
                            click.echo()
                            click.echo(
                                f"✗ Cannot decrypt existing secrets: {e}",
                                err=True,
                            )
                            click.echo(
                                "  The current master key may not match the key used to "
                                "encrypt secrets.",
                                err=True,
                            )
                            has_errors = True
                except ImportError:
                    pass
        except Exception as e:
            if verbose:
                click.echo()
                click.echo(f"Database check skipped: {e}", err=True)

    click.echo()
    if has_errors:
        click.echo("✗✗✗ Configuration has ERRORS ✗✗✗", err=True)
        sys.exit(2)
    elif has_warnings:
        click.echo("⚠️  Configuration has warnings", err=True)
        sys.exit(1)
    else:
        click.echo("✓ Configuration is valid and secure")
