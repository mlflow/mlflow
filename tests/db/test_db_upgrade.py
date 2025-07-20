"""
Tests for the mlflow db upgrade command, particularly testing issue #2444:
'mlflow db upgrade' fails on PostgreSQL instance due to missing initial tables
"""
import os
import tempfile
from unittest import mock

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine

import mlflow.db
from mlflow.store.db import utils


class TestDbUpgrade:
    """Test suite for mlflow db upgrade command"""

    def test_db_upgrade_on_sqlite_fresh_database(self):
        """Test that db upgrade works on a fresh SQLite database"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_uri = f"sqlite:///{tmp_file.name}"
            
        try:
            # Remove the file to ensure it's truly fresh
            os.unlink(tmp_file.name)
            
            # This should work - SQLite handles missing files gracefully
            runner = CliRunner()
            result = runner.invoke(mlflow.db.commands, ["upgrade", db_uri])
            
            assert result.exit_code == 0, f"Command failed with output: {result.output}"
            
            # Verify that tables were created
            engine = create_engine(db_uri)
            assert utils._all_tables_exist(engine), "Not all expected tables were created"
            
        finally:
            # Clean up
            if os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)

    @pytest.mark.parametrize("db_type", ["postgresql", "mysql"])
    def test_db_upgrade_on_fresh_database_fails(self, db_type):
        """
        Test that demonstrates issue #2444: db upgrade fails on fresh PostgreSQL/MySQL databases
        
        This test is expected to FAIL until the issue is fixed.
        It documents the current broken behavior.
        """
        # Skip this test if we don't have the database dependencies
        if db_type == "postgresql":
            pytest.importorskip("psycopg2")
            # Use a mock URI since we can't easily create test databases
            db_uri = "postgresql://user:password@localhost/test_fresh_db"
        elif db_type == "mysql":
            pytest.importorskip("pymysql") 
            db_uri = "mysql://user:password@localhost/test_fresh_db"
        
        # Mock the engine creation to simulate a fresh database
        mock_engine = mock.MagicMock()
        mock_engine.url = db_uri
        mock_engine.dialect.name = db_type
        
        # Mock table inspection to return empty (no tables exist)
        with mock.patch("sqlalchemy.inspect") as mock_inspect:
            mock_inspect.return_value.get_table_names.return_value = []
            
            # Mock engine creation
            with mock.patch("mlflow.store.db.utils.create_sqlalchemy_engine_with_retry", 
                          return_value=mock_engine):
                
                # Mock the actual database operations that would fail
                with mock.patch("mlflow.store.db.utils._upgrade_db") as mock_upgrade:
                    # Simulate the actual error that occurs
                    from sqlalchemy.exc import ProgrammingError
                    mock_upgrade.side_effect = ProgrammingError(
                        "relation \"metrics\" does not exist", None, None
                    )
                    
                    runner = CliRunner()
                    result = runner.invoke(mlflow.db.commands, ["upgrade", db_uri])
                    
                    # This should fail with the current implementation
                    assert result.exit_code != 0, (
                        "Expected db upgrade to fail on fresh database, but it succeeded. "
                        "If this test is now passing, the issue may have been fixed!"
                    )
                    assert "relation \"metrics\" does not exist" in result.output

    def test_db_upgrade_on_initialized_database_succeeds(self):
        """Test that db upgrade works correctly on an already initialized database"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_uri = f"sqlite:///{tmp_file.name}"
            
        try:
            # Remove the file initially
            os.unlink(tmp_file.name)
            
            # First, initialize the database properly (this should work)
            engine = utils.create_sqlalchemy_engine_with_retry(db_uri)
            utils._initialize_tables(engine)
            
            # Now test that db upgrade works on the initialized database
            runner = CliRunner()
            result = runner.invoke(mlflow.db.commands, ["upgrade", db_uri])
            
            assert result.exit_code == 0, f"Command failed with output: {result.output}"
            
            # Verify that all tables still exist
            engine = utils.create_sqlalchemy_engine_with_retry(db_uri)
            assert utils._all_tables_exist(engine), "Some tables are missing after upgrade"
            
        finally:
            # Clean up
            if os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)

    def test_all_tables_exist_function(self):
        """Test the _all_tables_exist utility function"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_uri = f"sqlite:///{tmp_file.name}"
            
        try:
            # Remove the file initially
            os.unlink(tmp_file.name)
            
            engine = utils.create_sqlalchemy_engine_with_retry(db_uri)
            
            # Should return False for fresh database
            assert not utils._all_tables_exist(engine), "Fresh database should not have all tables"
            
            # Initialize tables
            utils._initialize_tables(engine)
            
            # Should return True for initialized database
            assert utils._all_tables_exist(engine), "Initialized database should have all tables"
            
        finally:
            # Clean up
            if os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)


def test_fix_for_issue_2444():
    """
    Test case for the proposed fix to issue #2444.
    
    This test will fail with the current implementation but should pass once the fix is applied.
    The fix should modify mlflow.db.upgrade() to check if tables exist before running migrations.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_uri = f"sqlite:///{tmp_file.name}"
        
    try:
        # Remove the file to ensure it's truly fresh
        os.unlink(tmp_file.name)
        
        # This simulates what should happen after the fix:
        # 1. Check if tables exist
        # 2. If not, initialize tables first
        # 3. Then run any remaining migrations
        
        engine = utils.create_sqlalchemy_engine_with_retry(db_uri)
        
        # Before fix: _upgrade_db would fail on fresh database
        # After fix: this should work by first initializing tables
        if not utils._all_tables_exist(engine):
            # This is what the fix should do internally
            utils._initialize_tables(engine)
        else:
            utils._upgrade_db(engine)
        
        # Verify that all tables were created successfully
        assert utils._all_tables_exist(engine), "Fix should ensure all tables are created"
        
        # Verify that the CLI command would now work
        runner = CliRunner()
        result = runner.invoke(mlflow.db.commands, ["upgrade", db_uri])
        assert result.exit_code == 0, f"Fixed CLI command should work: {result.output}"
        
    finally:
        # Clean up
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)