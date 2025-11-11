import os

import click

from mlflow.exceptions import MlflowException
from mlflow.tracking import _get_store
from mlflow.utils.crypto import KEKManager, rotate_secret_encryption


@click.group("secrets", help="Commands for managing MLflow secrets.")
def commands():
    """MLflow secrets management CLI."""


@commands.command("rotate-kek", help="Rotate the KEK passphrase for all secrets.")
@click.option(
    "--new-passphrase",
    required=True,
    prompt=True,
    hide_input=True,
    confirmation_prompt=True,
    help="New KEK passphrase to use for encrypting secrets.",
)
@click.option(
    "--backend-store-uri",
    envvar="MLFLOW_BACKEND_STORE_URI",
    default=None,
    help="URI of the backend store. If not specified, uses MLFLOW_TRACKING_URI.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def rotate_kek(new_passphrase, backend_store_uri, yes):
    """
    Rotate the KEK passphrase for all secrets in the database.

    This command re-wraps all secret DEKs with a new KEK derived from the provided
    passphrase. The secret values themselves are not re-encrypted, making this
    operation efficient even for large numbers of secrets.

    CRITICAL: This CLI cannot set environment variables for your server. You MUST
    manually update BOTH environment variables in your deployment configuration:
    - MLFLOW_SECRETS_KEK_PASSPHRASE (to new passphrase)
    - MLFLOW_SECRETS_KEK_VERSION (incremented by 1)

    Failure to update both will cause decryption failures!

    Note that this operation requires the MLflow server to be shut down to ensure
    atomicity and prevent concurrent operations during rotation. The workflow is:

    1. Shut down the MLflow server
    2. Set MLFLOW_SECRETS_KEK_PASSPHRASE to the OLD passphrase (if not already set)
    3. Set MLFLOW_SECRETS_KEK_VERSION to the CURRENT version (if not already set)
    4. Run this command with the NEW passphrase
    5. Update your deployment config with BOTH new values:
       - MLFLOW_SECRETS_KEK_PASSPHRASE='new-passphrase'
       - MLFLOW_SECRETS_KEK_VERSION='<incremented>'
    6. Restart the MLflow server

    .. code-block:: bash

        # Step 1: Stop server
        $ systemctl stop mlflow-server

        # Step 2-3: Set current env vars (if needed)
        $ export MLFLOW_SECRETS_KEK_PASSPHRASE="old-passphrase"
        $ export MLFLOW_SECRETS_KEK_VERSION="1"

        # Step 4: Run rotation
        $ mlflow secrets rotate-kek --new-passphrase "new-passphrase"

        # Step 5: Update deployment config (example for Kubernetes)
        $ kubectl create secret generic mlflow-kek \\
            --from-literal=passphrase='new-passphrase' \\
            --from-literal=version='2' \\
            --dry-run=client -o yaml | kubectl apply -f -

        # Step 6: Restart server
        $ systemctl start mlflow-server
    """
    old_passphrase = os.getenv("MLFLOW_SECRETS_KEK_PASSPHRASE")
    if not old_passphrase:
        raise MlflowException(
            "MLFLOW_SECRETS_KEK_PASSPHRASE environment variable must be set to the "
            "current (old) passphrase before running KEK rotation.\n\n"
            "Example:\n"
            "  export MLFLOW_SECRETS_KEK_PASSPHRASE='current-passphrase'\n"
            "  export MLFLOW_SECRETS_KEK_VERSION='1'\n"
            "  mlflow secrets rotate-kek --new-passphrase 'new-passphrase'"
        )

    old_version = int(os.getenv("MLFLOW_SECRETS_KEK_VERSION", "1"))
    new_version = old_version + 1

    if not yes:
        click.echo("\n⚠️  WARNING: KEK Rotation Operation\n", err=True)
        click.echo("This operation will:", err=True)
        click.echo("  - Re-wrap all secret DEKs with a new KEK", err=True)
        click.echo(
            f"  - Update all secrets from kek_version {old_version} to {new_version}", err=True
        )
        click.echo("  - Require updating BOTH environment variables after completion:", err=True)
        click.echo("    * MLFLOW_SECRETS_KEK_PASSPHRASE='<new-passphrase>'", err=True)
        click.echo(f"    * MLFLOW_SECRETS_KEK_VERSION='{new_version}'\n", err=True)
        click.echo(
            "IMPORTANT: Ensure the MLflow server is shut down before proceeding.\n", err=True
        )

        if not click.confirm("Continue with KEK rotation?"):
            click.echo("KEK rotation cancelled.", err=True)
            return

    click.echo(f"Creating KEK managers (v{old_version} -> v{new_version})...")
    try:
        old_kek_manager = KEKManager(passphrase=old_passphrase, kek_version=old_version)
        new_kek_manager = KEKManager(passphrase=new_passphrase, kek_version=new_version)
    except Exception as e:
        raise MlflowException(f"Failed to create KEK managers: {e}") from e

    click.echo("Connecting to backend store...")
    try:
        store = _get_store(backend_store_uri)
    except Exception as e:
        raise MlflowException(f"Failed to connect to backend store: {e}") from e

    click.echo("Retrieving secrets to rotate...")
    try:
        from mlflow.store.tracking.dbmodels.models import SqlSecret

        with store.ManagedSessionMaker() as session:
            secrets = session.query(SqlSecret).filter(SqlSecret.kek_version == old_version).all()
            total_secrets = len(secrets)

            if total_secrets == 0:
                click.echo(
                    f"✓ No secrets found with kek_version={old_version}. Nothing to rotate.",
                    err=True,
                )
                return

            click.echo(f"Found {total_secrets} secrets to rotate.\n")

            rotated_count = 0

            with click.progressbar(
                secrets, label="Rotating secrets", show_pos=True, show_percent=True
            ) as progress:
                for secret in progress:
                    try:
                        result = rotate_secret_encryption(
                            secret.encrypted_value,
                            secret.wrapped_dek,
                            old_kek_manager,
                            new_kek_manager,
                        )

                        secret.wrapped_dek = result.wrapped_dek
                        secret.kek_version = new_version

                        rotated_count += 1

                    except Exception as e:
                        click.echo(f"\n✗ Failed to rotate secret {secret.secret_id}: {e}", err=True)
                        session.rollback()
                        raise MlflowException(
                            f"KEK rotation failed at secret {secret.secret_id}. "
                            "No changes were made. Fix the issue and re-run the command."
                        ) from e

            session.commit()

            click.echo(
                f"\n✓ Successfully rotated {rotated_count} secrets "
                f"from KEK v{old_version} to v{new_version}\n"
            )
            click.echo("=" * 80)
            click.echo("CRITICAL: Update BOTH environment variables in your deployment config:")
            click.echo("=" * 80)
            click.echo("\n  MLFLOW_SECRETS_KEK_PASSPHRASE='<new-passphrase>'")
            click.echo(f"  MLFLOW_SECRETS_KEK_VERSION='{new_version}'")
            click.echo("\nFailure to update BOTH variables will cause decryption failures!\n")
            click.echo("Platform-specific examples:\n")
            click.echo("Kubernetes:")
            click.echo("  $ kubectl create secret generic mlflow-kek \\")
            click.echo("      --from-literal=passphrase='<new-passphrase>' \\")
            click.echo(f"      --from-literal=version='{new_version}' \\")
            click.echo("      --dry-run=client -o yaml | kubectl apply -f -\n")
            click.echo("Docker Compose (.env file):")
            click.echo("  MLFLOW_SECRETS_KEK_PASSPHRASE=<new-passphrase>")
            click.echo(f"  MLFLOW_SECRETS_KEK_VERSION={new_version}\n")
            click.echo("Systemd (EnvironmentFile):")
            click.echo("  # Edit /etc/mlflow/secrets.env:")
            click.echo("  MLFLOW_SECRETS_KEK_PASSPHRASE=<new-passphrase>")
            click.echo(f"  MLFLOW_SECRETS_KEK_VERSION={new_version}\n")
            click.echo("After updating config:")
            click.echo("  $ systemctl restart mlflow-server  # or your restart command\n")

    except Exception as e:
        raise MlflowException(f"KEK rotation failed: {e}") from e
