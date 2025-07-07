#!/usr/bin/env python
"""
Debug script to understand tag count differences between MongoDB and SQLAlchemy stores.
"""

import sys
import os
import tempfile
import shutil
import uuid
from typing import List

# Ensure we're using Genesis-Flow
genesis_flow_path = "/Users/jagveersingh/Developer/autonomize/genesis/platform-agentops-mlops/genesis-flow"
if genesis_flow_path not in sys.path:
    sys.path.insert(0, genesis_flow_path)

from mlflow.entities import RunTag
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.mongodb_store import MongoDBStore
from mlflow.utils.time import get_current_time_millis

def debug_tag_differences():
    """Debug tag count differences between stores."""
    print("🔍 Debugging Tag Count Differences")
    print("=" * 50)
    
    # Set up stores
    temp_dir = tempfile.mkdtemp()
    sqlite_path = os.path.join(temp_dir, "test_debug.db")
    sql_uri = f"sqlite:///{sqlite_path}"
    sql_store = SqlAlchemyStore(sql_uri, "file:///tmp/artifacts")
    
    mongo_uri = "mongodb://localhost:27017/genesis_flow_debug"
    mongo_store = MongoDBStore(mongo_uri, "azure://artifacts")
    
    try:
        # Create experiments
        sql_exp_id = sql_store.create_experiment("debug_sql_exp")
        mongo_exp_id = mongo_store.create_experiment("debug_mongo_exp")
        
        # Create runs with no initial tags
        sql_run = sql_store.create_run(sql_exp_id, "test_user", get_current_time_millis(), [], "debug_run")
        mongo_run = mongo_store.create_run(mongo_exp_id, "test_user", get_current_time_millis(), [], "debug_run")
        
        sql_run_id = sql_run.info.run_id
        mongo_run_id = mongo_run.info.run_id
        
        print(f"Created SQL run: {sql_run_id}")
        print(f"Created MongoDB run: {mongo_run_id}")
        
        # Check initial tag counts
        sql_run_initial = sql_store.get_run(sql_run_id)
        mongo_run_initial = mongo_store.get_run(mongo_run_id)
        
        print(f"\n📊 Initial tag counts:")
        print(f"SQL store: {len(sql_run_initial.data.tags)} tags")
        print(f"MongoDB store: {len(mongo_run_initial.data.tags)} tags")
        
        print(f"\n📝 Initial SQL tags:")
        for tag in sql_run_initial.data.tags:
            if hasattr(tag, 'key'):
                print(f"  - {tag.key}: {tag.value}")
            else:
                print(f"  - {tag} (type: {type(tag)})")
        
        print(f"\n📝 Initial MongoDB tags:")
        for tag in mongo_run_initial.data.tags:
            if hasattr(tag, 'key'):
                print(f"  - {tag.key}: {tag.value}")
            else:
                print(f"  - {tag} (type: {type(tag)})")
        
        # Add same tags to both
        tags_to_add = [
            RunTag("environment", "production"),
            RunTag("model_type", "regression"),
            RunTag("version", "2.0")
        ]
        
        print(f"\n➕ Adding {len(tags_to_add)} tags to both stores...")
        for tag in tags_to_add:
            sql_store.set_tag(sql_run_id, tag)
            mongo_store.set_tag(mongo_run_id, tag)
        
        # Add batch tags
        batch_tags = [
            RunTag("data_source", "s3"),
            RunTag("experiment_type", "hyperparameter_tuning")
        ]
        
        print(f"➕ Adding {len(batch_tags)} batch tags...")
        sql_store.log_batch(sql_run_id, [], [], batch_tags)
        mongo_store.log_batch(mongo_run_id, [], [], batch_tags)
        
        # Check final tag counts
        sql_run_final = sql_store.get_run(sql_run_id)
        mongo_run_final = mongo_store.get_run(mongo_run_id)
        
        print(f"\n📊 Final tag counts:")
        print(f"SQL store: {len(sql_run_final.data.tags)} tags")
        print(f"MongoDB store: {len(mongo_run_final.data.tags)} tags")
        
        print(f"\n📝 Final SQL tags:")
        for tag in sorted(sql_run_final.data.tags, key=lambda t: t.key if hasattr(t, 'key') else str(t)):
            if hasattr(tag, 'key'):
                print(f"  - {tag.key}: {tag.value}")
            else:
                print(f"  - {tag} (type: {type(tag)})")
        
        print(f"\n📝 Final MongoDB tags:")
        for tag in sorted(mongo_run_final.data.tags, key=lambda t: t.key if hasattr(t, 'key') else str(t)):
            if hasattr(tag, 'key'):
                print(f"  - {tag.key}: {tag.value}")
            else:
                print(f"  - {tag} (type: {type(tag)})")
        
        # Find differences
        sql_tag_dict = {tag.key: tag.value for tag in sql_run_final.data.tags if hasattr(tag, 'key')}
        mongo_tag_dict = {tag.key: tag.value for tag in mongo_run_final.data.tags if hasattr(tag, 'key')}
        
        sql_only = set(sql_tag_dict.keys()) - set(mongo_tag_dict.keys())
        mongo_only = set(mongo_tag_dict.keys()) - set(sql_tag_dict.keys())
        
        print(f"\n🔍 Differences:")
        if sql_only:
            print(f"Tags only in SQL store: {sql_only}")
        if mongo_only:
            print(f"Tags only in MongoDB store: {mongo_only}")
        
        if not sql_only and not mongo_only:
            print("No key differences - values might differ")
            for key in sql_tag_dict:
                if key in mongo_tag_dict and sql_tag_dict[key] != mongo_tag_dict[key]:
                    print(f"Value diff for {key}: SQL='{sql_tag_dict[key]}' vs MongoDB='{mongo_tag_dict[key]}'")
        
    finally:
        # Cleanup
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if hasattr(mongo_store, 'sync_client'):
            mongo_store.sync_client.drop_database("genesis_flow_debug")

if __name__ == "__main__":
    debug_tag_differences()