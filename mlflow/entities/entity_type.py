"""
Entity type constants for MLflow's entity_association table.
The entity_association table enables many-to-many relationships between different
MLflow entities. It uses source and destination type/id pairs to create flexible
associations without requiring dedicated junction tables for each relationship type.
"""


class EntityAssociationType:
    """Constants for entity types used in the entity_association table."""

    EXPERIMENT = "experiment"
    EVALUATION_DATASET = "evaluation_dataset"
    RUN = "run"
    MODEL = "model"
    TRACE = "trace"
    PROMPT_VERSION = "prompt_version"
