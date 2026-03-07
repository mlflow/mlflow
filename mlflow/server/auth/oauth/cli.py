import click
import sqlalchemy
from sqlalchemy.orm import Session


@click.group(name="oauth")
def commands():
    pass


@commands.command(name="rotate-key")
@click.option("--url", required=True, help="Database URI")
@click.option("--old-key", required=True, help="Old encryption key (64 hex chars)")
@click.option("--new-key", required=True, help="New encryption key (64 hex chars)")
def rotate_key(url: str, old_key: str, new_key: str) -> None:
    from mlflow.server.auth.oauth.db.models import SqlSession
    from mlflow.server.auth.oauth.session import decrypt_token, encrypt_token

    engine = sqlalchemy.create_engine(url)
    with Session(engine) as db:
        sessions = db.query(SqlSession).all()
        count = 0
        for session in sessions:
            changed = False
            if session.access_token_enc:
                plaintext = decrypt_token(session.access_token_enc, old_key)
                session.access_token_enc = encrypt_token(plaintext, new_key)
                changed = True
            if session.refresh_token_enc:
                plaintext = decrypt_token(session.refresh_token_enc, old_key)
                session.refresh_token_enc = encrypt_token(plaintext, new_key)
                changed = True
            if changed:
                count += 1
        db.commit()
        click.echo(f"Rotated encryption key for {count} sessions.")
    engine.dispose()
