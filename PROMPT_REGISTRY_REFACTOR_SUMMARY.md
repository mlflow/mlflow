# Prompt Registry Refactor Summary

## Overview
This document summarizes the implementation of the prompt registry refactor following the TL's guidance to move prompt registry APIs into the Model Registry AbstractStore.

## Branch
- Branch name: `prompt_registry_refactor`
- Based on: main branch

## Key Changes

### 1. AbstractStore Enhancement (`mlflow/store/model_registry/abstract_store.py`)
Added prompt-specific methods with default implementations that use the existing RegisteredModel/ModelVersion infrastructure:
- `create_prompt()` - Creates a prompt using RegisteredModel with special tags
- `get_prompt()` - Retrieves a prompt by name and version/alias
- `search_prompts()` - Searches for prompts using RegisteredModel search with filters
- `delete_prompt()` - Deletes a prompt (RegisteredModel)
- `create_prompt_version()` - Creates a new prompt version (ModelVersion)
- `get_prompt_version()` - Gets a specific prompt version
- `delete_prompt_version()` - Deletes a prompt version
- `set_prompt_tag()` / `delete_prompt_tag()` - Manages prompt-level tags
- `set_prompt_version_tag()` / `delete_prompt_version_tag()` - Manages version-level tags
- `set_prompt_alias()` / `delete_prompt_alias()` / `get_prompt_version_by_alias()` - Manages aliases

### 2. ModelRegistryClient Updates (`mlflow/tracking/_model_registry/client.py`)
Added corresponding prompt methods that simply delegate to the store:
```python
def create_prompt(self, name, template, description=None, tags=None):
    return self.store.create_prompt(name, template, description, tags)
# ... similar for all other prompt methods
```

### 3. MlflowClient Simplification (`mlflow/tracking/client.py`)
- Simplified prompt methods to delegate to registry client
- Removed UC blocking logic from client methods
- Added `_parse_prompt_uri()` helper function
- Added `set_terminated()` method that was missing

### 4. UC Store Overrides (`mlflow/store/_unity_catalog/registry/rest_store.py`)
Added placeholder prompt method overrides in UcModelRegistryStore:
```python
def create_prompt(self, name, template, description=None, tags=None):
    if not MLFLOW_ENABLE_UC_PROMPT_SUPPORT.get():
        raise MlflowException("UC prompt support is not enabled...")
    # TODO: Call UC prompt endpoints when available
    return super().create_prompt(name, template, description, tags)
# ... similar for all other prompt methods
```

### 5. Feature Flag Support
- Added `MLFLOW_ENABLE_UC_PROMPT_SUPPORT` environment variable (`mlflow/environment_variables.py`)
- Modified `is_prompt_supported_registry()` to support UC when flag is enabled
- UC registries (databricks-uc and uc:) now support prompts when flag is set

### 6. Bug Fixes
- Fixed `add_prompt_filter_string()` function call to use correct parameter name (`is_prompt` instead of `include_prompts`)
- Fixed RegisteredModel tags handling (they're stored as a dictionary, not a list)
- Added missing imports and fixed circular dependencies

## Architecture Benefits

1. **Clean Separation**: Prompt logic is now in the store layer, not scattered across client classes
2. **Easy UC Integration**: UC can simply override the prompt methods in UcModelRegistryStore
3. **Backward Compatibility**: OSS workflow remains unchanged
4. **Feature Flag Control**: UC prompt support can be enabled/disabled via environment variable
5. **Consistent Pattern**: Follows the same pattern as other model registry operations

## Testing

Created comprehensive test script (`test_prompt_refactor.py`) that validates:
- OSS prompt registry functionality (create, load, version, alias, search, log to run)
- UC registry blocking without feature flag
- UC registry enablement with feature flag

All tests pass successfully.

## Next Steps

When UC prompt endpoints are available:
1. Implement the actual UC API calls in the overridden methods in `UcModelRegistryStore`
2. Remove the `super()` calls and implement proper UC-specific logic
3. Handle UC-specific features like three-tier naming (catalog.schema.prompt_name)
4. Add proper error handling and UC-specific validations

## Migration Path

For existing users:
1. OSS users: No changes needed, everything works as before
2. UC users: Set `MLFLOW_ENABLE_UC_PROMPT_SUPPORT=true` when UC endpoints are ready
3. Legacy Databricks workspace users: Continue to be blocked from prompt operations 