"""
Entity type constants for MLflow's entity_association table.

The entity_association table enables many-to-many relationships between different
MLflow entities. It uses source and destination type/id pairs to create flexible
associations without requiring dedicated junction tables for each relationship type.

How entity associations work:
1. Each association has a source entity (source_type, source_id) and destination
   entity (destination_type, destination_id)
2. The composite primary key ensures unique associations
3. Bidirectional indexes enable efficient queries from both directions

The entity_association table schema:
- source_type: One of the EntityType constants below
- source_id: The UUID of the source entity
- destination_type: One of the EntityType constants below
- destination_id: The UUID of the destination entity
- association_id: A unique identifier for the association
- created_time: Timestamp when the association was created

Example usage in SQLAlchemy:
    association = SqlEntityAssociation(
        association_id=_generate_uuid(),
        source_type=EntityType.EVALUATION_DATASET,
        source_id=dataset_id,
        destination_type=EntityType.EXPERIMENT,
        destination_id=experiment_id,
        created_time=get_current_time_millis()
    )
"""


class EntityAssociationType:
    """Constants for entity types used in the entity_association table."""

    EXPERIMENT = "experiment"
    EVALUATION_DATASET = "evaluation_dataset"
    RUN = "run"
    MODEL = "model"
