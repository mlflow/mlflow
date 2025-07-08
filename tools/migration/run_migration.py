#!/usr/bin/env python
"""
Simple Migration Runner with Configuration File Support

This script provides an easy way to run the PostgreSQL to MongoDB migration
using configuration files or direct command line arguments.

Usage:
    # Using configuration file
    python tools/migration/run_migration.py --config migration_config.json
    
    # Direct command line (for your specific case)
    python tools/migration/run_migration.py --quick-migrate-autonomize
    
    # Custom URIs
    python tools/migration/run_migration.py \\
        --postgres-uri "postgresql://user:pass@host:port/db" \\
        --mongodb-uri "mongodb://localhost:27017/mlflow_migrated"
"""

import argparse
import json
import sys
import os

def load_config(config_file: str) -> dict:
    """Load migration configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        sys.exit(1)

def run_migration_with_config(config: dict, dry_run: bool = False):
    """Run migration using configuration dictionary."""
    postgres_uri = config['source']['connection_uri']
    mongodb_uri = config['destination']['recommended']
    batch_size = config['migration_settings'].get('batch_size', 1000)
    
    # Import and run the migrator
    from postgres_to_mongodb import PostgreSQLToMongoDBMigrator
    
    migrator = PostgreSQLToMongoDBMigrator(
        postgres_uri=postgres_uri,
        mongodb_uri=mongodb_uri,
        batch_size=batch_size,
        dry_run=dry_run
    )
    
    return migrator.run_migration()

def main():
    parser = argparse.ArgumentParser(
        description="Easy MLflow PostgreSQL to MongoDB Migration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Migration Examples:

1. Migrate from Autonomize PostgreSQL to local MongoDB:
   python tools/migration/run_migration.py --quick-migrate-autonomize

2. Dry run to analyze data first:
   python tools/migration/run_migration.py --quick-migrate-autonomize --dry-run

3. Migrate to Azure Cosmos DB:
   python tools/migration/run_migration.py \\
     --postgres-uri "postgresql://postgres:7HrX26sHIZz8yffytPc0@autonomize-database-1002.cwpqzu4drrfr.us-east-1.rds.amazonaws.com:5432/mlflow" \\
     --mongodb-uri "mongodb://account:key@account.mongo.cosmos.azure.com:10255/mlflow?ssl=true"

4. Using configuration file:
   python tools/migration/run_migration.py --config migration_config.json
        """
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        help='Path to JSON configuration file'
    )
    
    # Quick migration option for Autonomize
    parser.add_argument(
        '--quick-migrate-autonomize',
        action='store_true',
        help='Quick migration from Autonomize PostgreSQL to local MongoDB'
    )
    
    # Direct URI options
    parser.add_argument(
        '--postgres-uri',
        help='PostgreSQL connection URI'
    )
    
    parser.add_argument(
        '--mongodb-uri',
        help='MongoDB connection URI'
    )
    
    # Migration options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for migration (default: 1000)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze data without performing migration'
    )
    
    args = parser.parse_args()
    
    # Determine migration parameters
    if args.quick_migrate_autonomize:
        # Quick migration from Autonomize PostgreSQL to local MongoDB
        print("🚀 Quick Migration: Autonomize PostgreSQL → Local MongoDB")
        print("=" * 60)
        
        postgres_uri = "postgresql://postgres:7HrX26sHIZz8yffytPc0@autonomize-database-1002.cwpqzu4drrfr.us-east-1.rds.amazonaws.com:5432/mlflow"
        mongodb_uri = "mongodb://localhost:27017/mlflow_migrated_from_autonomize"
        
        print(f"Source:      {postgres_uri}")
        print(f"Destination: {mongodb_uri}")
        print(f"Batch Size:  {args.batch_size}")
        print(f"Dry Run:     {args.dry_run}")
        print()
        
        if not args.dry_run:
            confirm = input("Proceed with migration? (y/N): ")
            if confirm.lower() != 'y':
                print("Migration cancelled.")
                sys.exit(0)
        
        # Import and run the migrator
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from postgres_to_mongodb import PostgreSQLToMongoDBMigrator
        
        migrator = PostgreSQLToMongoDBMigrator(
            postgres_uri=postgres_uri,
            mongodb_uri=mongodb_uri,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
        
        success = migrator.run_migration()
        
    elif args.config:
        # Use configuration file
        print(f"📄 Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        print("Configuration loaded successfully:")
        print(f"Source:      {config['source']['connection_uri']}")
        print(f"Destination: {config['destination']['recommended']}")
        print()
        
        success = run_migration_with_config(config, dry_run=args.dry_run)
        
    elif args.postgres_uri and args.mongodb_uri:
        # Use direct URIs
        print("🔗 Direct URI Migration")
        print("=" * 30)
        print(f"Source:      {args.postgres_uri}")
        print(f"Destination: {args.mongodb_uri}")
        print()
        
        # Import and run the migrator
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from postgres_to_mongodb import PostgreSQLToMongoDBMigrator
        
        migrator = PostgreSQLToMongoDBMigrator(
            postgres_uri=args.postgres_uri,
            mongodb_uri=args.mongodb_uri,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
        
        success = migrator.run_migration()
        
    else:
        print("❌ Error: Must specify either --quick-migrate-autonomize, --config, or both --postgres-uri and --mongodb-uri")
        parser.print_help()
        sys.exit(1)
    
    # Report results
    if success:
        print("\n🎉 Migration completed successfully!")
        
        if not args.dry_run:
            print("\n📋 Next Steps:")
            print("1. Verify data integrity using Genesis-Flow")
            print("2. Update your application to use the new MongoDB URI")
            print("3. Test all MLflow operations with the new backend")
            print("4. Consider updating artifact storage configuration")
            
            if args.quick_migrate_autonomize:
                print(f"\n🔧 Update your MLflow configuration:")
                print(f"   mlflow.set_tracking_uri('{mongodb_uri}')")
                print(f"   mlflow.set_registry_uri('{mongodb_uri}')")
        
        sys.exit(0)
    else:
        print("\n❌ Migration failed - check logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()