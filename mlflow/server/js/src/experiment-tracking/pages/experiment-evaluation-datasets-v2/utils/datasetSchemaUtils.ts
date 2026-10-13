import type { DatasetRecord } from '../hooks/useDatasetsQueries';
type DatasetSchemaType = 'singleturn' | 'multiturn';
const REQUIRED_MULTITURN_INPUT_FIELDS = new Set(['goal']);
const ALLOWED_MULTITURN_INPUT_FIELDS = new Set(['goal', 'persona', 'context', 'simulation_guidelines']);

/**
 * Returns default record values depending on the schema type.
 */
export function getDefaultRecord(
  records: DatasetRecord[],
  editedRows: Record<string, Partial<DatasetRecord>> = {},
): Partial<DatasetRecord> {
  const schemaType = detectDatasetSchema(records, editedRows);
  if (schemaType === 'multiturn') {
    return {
      inputs: {
        goal: 'Complete the user request',
        persona: 'Helpful assistant',
      },
      expectations: { guidelines: ['The response must be professional'] },
    };
  }
  return {
    inputs: { messages: [{ role: 'user', content: 'Hello' }] },
    expectations: { guidelines: ['The response must be professional'] },
  };
}

/**
 * Validates all records have consistent schema.
 * Throws an error if validation fails.
 */
export function validateSchemaConsistency(
  records: DatasetRecord[],
  editedRows: Record<string, Partial<DatasetRecord>> = {},
) {
  if (records.length === 0) return;

  const mergedRecordInputs = records.map((record) => {
    const editedData = editedRows[record.dataset_record_id];
    return editedData?.inputs ?? record.inputs;
  });

  const recordsWithGoal = mergedRecordInputs.filter((inputs) =>
    [...REQUIRED_MULTITURN_INPUT_FIELDS].every((field) => field in (inputs || {})),
  );
  const allMultiturn = recordsWithGoal.length === mergedRecordInputs.length;
  const allSingleturn = recordsWithGoal.length === 0;

  if (allMultiturn) {
    validateMultiturnFields(records, editedRows);
    return;
  }

  if (allSingleturn) {
    return;
  }

  const multiturnCount = recordsWithGoal.length;
  const singleturnCount = mergedRecordInputs.length - multiturnCount;

  throw new Error(
    `Mixed schemas: ${multiturnCount} record(s) have 'goal' field (multi-turn schema), ${singleturnCount} record(s) do not (single-turn schema). All records in the same dataset must use the same schema.`,
  );
}

/**
 * Detects schema type from records.
 * If any record has a 'goal' field in inputs, it's multiturn.
 * Returns 'singleturn' as default for empty datasets.
 */
function detectDatasetSchema(
  records: DatasetRecord[],
  editedRows: Record<string, Partial<DatasetRecord>> = {},
): DatasetSchemaType {
  const isMultiturn = records.some((record) => {
    const inputs = editedRows[record.dataset_record_id]?.inputs ?? record.inputs;
    return [...REQUIRED_MULTITURN_INPUT_FIELDS].every((field) => field in (inputs || {}));
  });
  return isMultiturn ? 'multiturn' : 'singleturn';
}

/**
 * Validates that multiturn records only contain allowed input fields (goal, persona, context, simulation_guidelines).
 * Throws an error if invalid fields are found.
 */
function validateMultiturnFields(records: DatasetRecord[], editedRows: Record<string, Partial<DatasetRecord>>) {
  for (const record of records) {
    const inputs = editedRows[record.dataset_record_id]?.inputs ?? record.inputs;
    const invalidFields = Object.keys(inputs || {}).filter((field) => !ALLOWED_MULTITURN_INPUT_FIELDS.has(field));

    if (invalidFields.length > 0) {
      throw new Error(
        `Invalid field(s) in multiturn record: '${invalidFields.join("', '")}'. Allowed fields are: ${[...ALLOWED_MULTITURN_INPUT_FIELDS].join(', ')}.`,
      );
    }
  }
}
