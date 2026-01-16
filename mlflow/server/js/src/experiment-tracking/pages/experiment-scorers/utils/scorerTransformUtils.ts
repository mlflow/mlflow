import { type CausableError, ErrorName, PredefinedError } from '@databricks/web-shared/errors';
import { ErrorLogType } from '@databricks/web-shared/errors';
import type {
  ScheduledScorer,
  LLMScorer,
  CustomCodeScorer,
  ScorerConfig,
  LLMTemplate,
  JudgeOutputTypeSpec,
  JudgePrimitiveOutputType,
} from '../types';
import { LLM_TEMPLATE, isGuidelinesTemplate } from '../types';

const PRIMITIVE_TO_JSON_SCHEMA: Record<JudgePrimitiveOutputType, string> = {
  bool: 'boolean',
  int: 'integer',
  float: 'number',
  str: 'string',
};

const JSON_SCHEMA_TO_PRIMITIVE: Record<string, JudgePrimitiveOutputType> = {
  boolean: 'bool',
  integer: 'int',
  number: 'float',
  string: 'str',
};

/**
 * Convert JudgeOutputTypeSpec to JSON Schema format for API serialization.
 * Matches the format expected by InstructionsJudge._deserialize_feedback_value_type
 * in mlflow/genai/judges/instructions_judge/__init__.py
 */
function outputTypeSpecToJsonSchema(spec: JudgeOutputTypeSpec | undefined): Record<string, unknown> | undefined {
  if (!spec) {
    return undefined;
  }

  switch (spec.kind) {
    case 'bool':
    case 'int':
    case 'float':
    case 'str':
      return { type: PRIMITIVE_TO_JSON_SCHEMA[spec.kind] };

    case 'categorical':
      if (!spec.categoricalOptions || spec.categoricalOptions.length === 0) {
        return undefined;
      }
      return {
        type: 'string',
        enum: spec.categoricalOptions,
      };

    case 'dict':
      return {
        type: 'object',
        additionalProperties: {
          type: PRIMITIVE_TO_JSON_SCHEMA[spec.dictValueType || 'int'],
        },
      };

    case 'list':
      return {
        type: 'array',
        items: {
          type: PRIMITIVE_TO_JSON_SCHEMA[spec.listElementType || 'str'],
        },
      };

    default:
      return undefined;
  }
}

function formDataToOutputTypeSpec(formData: LLMScorerFormData): JudgeOutputTypeSpec | undefined {
  const kind = formData.outputTypeKind;
  if (!kind || kind === 'default') {
    return undefined;
  }

  return {
    kind,
    categoricalOptions: formData.categoricalOptions
      ?.split('\n')
      .map((line) => line.trim())
      .filter(Boolean),
    dictValueType: formData.dictValueType,
    listElementType: formData.listElementType,
  };
}

export function outputTypeSpecToFormData(
  spec: JudgeOutputTypeSpec | undefined,
): Pick<LLMScorerFormData, 'outputTypeKind' | 'categoricalOptions' | 'dictValueType' | 'listElementType'> {
  if (!spec) {
    return { outputTypeKind: 'default' };
  }

  return {
    outputTypeKind: spec.kind,
    categoricalOptions: spec.categoricalOptions?.join('\n'),
    dictValueType: spec.dictValueType,
    listElementType: spec.listElementType,
  };
}

/**
 * Convert JSON Schema format back to JudgeOutputTypeSpec for loading from API.
 * Reverse of outputTypeSpecToJsonSchema.
 */
function jsonSchemaToOutputTypeSpec(schema: Record<string, unknown> | undefined): JudgeOutputTypeSpec | undefined {
  if (!schema) {
    return undefined;
  }

  // Check for enum (Literal/categorical type)
  if (Array.isArray(schema['enum'])) {
    return {
      kind: 'categorical',
      categoricalOptions: schema['enum'] as string[],
    };
  }

  const schemaType = schema['type'];
  if (typeof schemaType !== 'string') {
    return undefined;
  }

  // Check for primitive types
  if (schemaType in JSON_SCHEMA_TO_PRIMITIVE) {
    return {
      kind: JSON_SCHEMA_TO_PRIMITIVE[schemaType],
    };
  }

  // Check for object (dict) type
  if (schemaType === 'object') {
    const additionalProps = schema['additionalProperties'] as Record<string, unknown> | undefined;
    if (additionalProps && typeof additionalProps['type'] === 'string') {
      return {
        kind: 'dict',
        dictValueType: JSON_SCHEMA_TO_PRIMITIVE[additionalProps['type']] || 'int',
      };
    }
    return { kind: 'dict', dictValueType: 'int' };
  }

  // Check for array (list) type
  if (schemaType === 'array') {
    const items = schema['items'] as Record<string, unknown> | undefined;
    if (items && typeof items['type'] === 'string') {
      return {
        kind: 'list',
        listElementType: JSON_SCHEMA_TO_PRIMITIVE[items['type']] || 'str',
      };
    }
    return { kind: 'list', listElementType: 'str' };
  }

  return undefined;
}
import type { LLMScorerFormData } from '../LLMScorerFormRenderer';
import type { CustomCodeScorerFormData } from '../CustomCodeScorerFormRenderer';
import { ScorerEvaluationScope, type ScorerType } from '../constants';
import type { RegisterScorerResponse, MLflowScorer } from '../api';
import { isEvaluatingSessionsInScorersEnabled } from '../../../../common/utils/FeatureUtils';
import { isUndefined } from 'lodash';

// Union type for all form data - combines both form interfaces
export type ScorerFormData = (LLMScorerFormData | CustomCodeScorerFormData) & {
  scorerType: ScorerType;
  evaluationScope?: ScorerEvaluationScope;
};

// Local error class for scorer transformation issues
export class ScorerTransformationError extends PredefinedError {
  errorLogType = ErrorLogType.ApplicationError;
  errorName = ErrorName.ScorerTransformationError;
  isUserError = true;
  displayMessage: React.ReactNode;

  constructor(message?: string, cause?: CausableError) {
    super(message, cause);
    this.displayMessage = this.message;
  }
}

/**
 * Transform backend ScorerConfig to frontend ScheduledScorer
 */
export function transformScorerConfig(config: ScorerConfig): ScheduledScorer {
  const baseFields: Partial<ScheduledScorer> = {
    name: config.name,
    // Convert from backend float (0-1) to frontend percentage (0-100)
    sampleRate: config.sample_rate !== undefined ? config.sample_rate * 100 : undefined,
    version: config.scorer_version,
    disableMonitoring: false,
  };

  // Only add filterString if it has a value
  if (config.filter_string) {
    baseFields.filterString = config.filter_string;
  }

  try {
    const serializedData = JSON.parse(config.serialized_scorer);

    if (isEvaluatingSessionsInScorersEnabled() && serializedData.is_session_level_scorer) {
      baseFields.isSessionLevelScorer = true;
    }

    // Determine scorer type based on the serialized data
    if (serializedData.instructions_judge_pydantic_data) {
      // Instructions-based LLM scorer
      const instructions = serializedData.instructions_judge_pydantic_data.instructions || '';
      const model = serializedData.instructions_judge_pydantic_data.model;
      const outputType = jsonSchemaToOutputTypeSpec(
        serializedData.instructions_judge_pydantic_data.feedback_value_type,
      );
      const result = {
        ...baseFields,
        type: 'llm',
        llmTemplate: LLM_TEMPLATE.CUSTOM,
        instructions,
        model,
        is_instructions_judge: true,
        outputType,
      } as LLMScorer;
      return result;
    } else if (isGuidelinesTemplate(serializedData.builtin_scorer_class)) {
      const rawGuidelines = serializedData.builtin_scorer_pydantic_data?.guidelines || [];
      // Ensure guidelines is always an array - if it's a string, put it in an array
      const guidelines = Array.isArray(rawGuidelines) ? rawGuidelines : [rawGuidelines].filter(Boolean);
      const model = serializedData.builtin_scorer_pydantic_data?.model;
      return {
        ...baseFields,
        type: 'llm',
        llmTemplate: serializedData.builtin_scorer_class,
        guidelines,
        model,
        is_instructions_judge: false,
      } as LLMScorer;
    } else if (serializedData.builtin_scorer_class && config.builtin) {
      const model = serializedData.builtin_scorer_pydantic_data?.model;
      return {
        ...baseFields,
        type: 'llm',
        llmTemplate: serializedData.builtin_scorer_class,
        model,
        is_instructions_judge: false,
      } as LLMScorer;
    } else {
      // Custom scorer - extract code from call_source
      const callSource = serializedData.call_source || '';
      const originalFuncName = serializedData.original_func_name;
      const callSignature = serializedData.call_signature;

      let code;
      if (originalFuncName && callSignature) {
        // Build complete function definition with name and signature
        code = `def ${originalFuncName}${callSignature}:\n    ${callSource.replace(/\n/g, '\n    ')}`;
      } else {
        // Just use the call_source as-is
        code = callSource;
      }

      return {
        ...baseFields,
        type: 'custom-code',
        code,
        callSignature,
        originalFuncName,
      } as CustomCodeScorer;
    }
  } catch (error) {
    const cause = error instanceof Error ? error : new Error(String(error));
    throw new ScorerTransformationError(`Failed to parse scorer configuration: ${cause.message}`, cause);
  }
}

/**
 * Transform frontend ScheduledScorer to backend ScorerConfig
 */
export function transformScheduledScorer(scorer: ScheduledScorer): ScorerConfig {
  const config: ScorerConfig = {
    name: scorer.name,
    serialized_scorer: '',
  };

  // Add sample_rate if provided (convert from percentage 0-100 to float 0-1)
  if (scorer.sampleRate !== undefined) {
    config.sample_rate = scorer.sampleRate / 100;
  }

  // Add filter_string if provided
  if (scorer.filterString) {
    config.filter_string = scorer.filterString;
  }

  // Common base for all serialized scorers
  const baseSerializedScorer: {
    mlflow_version: string;
    serialization_version: number;
    is_session_level_scorer?: boolean;
  } = {
    mlflow_version: '3.3.2+ui', // Valid PyPI version with local version identifier to distinguish scorers created from UI
    serialization_version: 1,
  };

  if (isEvaluatingSessionsInScorersEnabled() && !isUndefined(scorer.isSessionLevelScorer)) {
    baseSerializedScorer.is_session_level_scorer = scorer.isSessionLevelScorer;
  }

  // Build serialized_scorer based on scorer type
  if (scorer.type === 'llm') {
    const llmScorer = scorer as LLMScorer;

    if (llmScorer.is_instructions_judge) {
      if (!llmScorer.instructions) {
        throw new ScorerTransformationError('Instructions are required for instructions-based LLM scorers');
      }
      const feedbackValueType = outputTypeSpecToJsonSchema(llmScorer.outputType);
      config.serialized_scorer = JSON.stringify({
        ...baseSerializedScorer,
        name: llmScorer.name,
        aggregations: [],
        builtin_scorer_class: null,
        builtin_scorer_pydantic_data: null,
        call_source: null,
        call_signature: null,
        original_func_name: null,
        instructions_judge_pydantic_data: {
          instructions: llmScorer.instructions || '',
          ...(llmScorer.model && { model: llmScorer.model }),
          ...(feedbackValueType && { feedback_value_type: feedbackValueType }),
        },
      });
      config.custom = {};
    } else if (llmScorer.llmTemplate) {
      // Build pydantic data - common fields for all built-in scorers
      const pydanticData: any = {
        name: llmScorer.name,
        required_columns: ['outputs', 'inputs'],
      };

      // Add guidelines if this is a Guidelines or ConversationalGuidelines scorer
      if (isGuidelinesTemplate(llmScorer.llmTemplate) && llmScorer.guidelines) {
        pydanticData.guidelines = llmScorer.guidelines;
      }

      // Add model if specified
      if (llmScorer.model) {
        pydanticData.model = llmScorer.model;
      }

      config.serialized_scorer = JSON.stringify({
        ...baseSerializedScorer,
        name: llmScorer.name,
        builtin_scorer_class: llmScorer.llmTemplate,
        builtin_scorer_pydantic_data: pydanticData,
      });
      config.builtin = {
        name: llmScorer.name,
      };
    }
  } else if (scorer.type === 'custom-code') {
    const customCodeScorer = scorer as CustomCodeScorer;

    // For custom code scorers, we preserve the original serialized format
    // and only update the modifiable fields (sample_rate and filter_string)
    // The serialized_scorer should remain unchanged from the original
    config.serialized_scorer = JSON.stringify({
      ...baseSerializedScorer,
      name: customCodeScorer.name,
      call_source: customCodeScorer.code,
      call_signature: customCodeScorer.callSignature,
      original_func_name: customCodeScorer.originalFuncName,
    });
    config.custom = {}; // this is needed for custom scorers
  }

  return config;
}

/**
 * Convert registerScorer API response to ScorerConfig
 * Maps the response fields from the /api/3.0/mlflow/scorers/register endpoint to the ScorerConfig type.
 */
export function convertRegisterScorerResponseToConfig(response: RegisterScorerResponse): ScorerConfig {
  const config: ScorerConfig = {
    name: response.name,
    serialized_scorer: response.serialized_scorer,
    scorer_version: response.version,
  };

  // Check if this is a built-in scorer by parsing serialized_scorer
  try {
    const serializedData = JSON.parse(response.serialized_scorer);
    if (serializedData.builtin_scorer_class) {
      config.builtin = { name: response.name };
    }
  } catch {
    // If parsing fails, treat as non-builtin scorer
  }

  return config;
}

/**
 * Convert MLflowScorer from listScorers API response to ScorerConfig
 * Maps the response fields from the /api/3.0/mlflow/scorers/list endpoint to the ScorerConfig type.
 */
export function convertMLflowScorerToConfig(scorer: MLflowScorer): ScorerConfig {
  const config: ScorerConfig = {
    name: scorer.scorer_name,
    serialized_scorer: scorer.serialized_scorer,
    scorer_version: scorer.scorer_version,
  };

  // Check if this is a built-in scorer by parsing serialized_scorer
  try {
    const serializedData = JSON.parse(scorer.serialized_scorer);
    if (serializedData.builtin_scorer_class) {
      config.builtin = { name: scorer.scorer_name };
    }
  } catch {
    // If parsing fails, treat as non-builtin scorer
  }

  return config;
}

/**
 * Convert ScorerFormData to ScheduledScorer for API calls
 * Supports both creating new scorers and updating existing ones
 */
export function convertFormDataToScheduledScorer(
  formData: ScorerFormData,
  baseScorer?: ScheduledScorer,
): ScheduledScorer {
  // For new scorers, create base object from scratch
  if (!baseScorer) {
    const newScorer: ScheduledScorer = {
      name: formData.name,
      sampleRate: formData.sampleRate,
      filterString: formData.filterString || '',
      type: formData.scorerType,
      isSessionLevelScorer: formData.evaluationScope === ScorerEvaluationScope.SESSIONS,
    } as ScheduledScorer;

    if (formData.scorerType === 'llm') {
      const llmFormData = formData as LLMScorerFormData;
      const result = {
        ...newScorer,
        type: 'llm' as const,
        llmTemplate: llmFormData.llmTemplate as LLMTemplate,
        // Add guidelines if this is a Guidelines or ConversationalGuidelines scorer
        guidelines: isGuidelinesTemplate(llmFormData.llmTemplate)
          ? (llmFormData.guidelines || '')
              .split('\n')
              .map((line) => line.trim())
              .filter(Boolean)
          : undefined,
        // Add instructions for instructions judges (Custom, Safety, RelevanceToQuery)
        instructions: llmFormData.isInstructionsJudge ? llmFormData.instructions : undefined,
        // Add model for all LLM scorers
        model: llmFormData.model || undefined,
        is_instructions_judge: llmFormData.isInstructionsJudge,
        outputType: llmFormData.isInstructionsJudge ? formDataToOutputTypeSpec(llmFormData) : undefined,
      };
      return result;
    }
  }

  // For updating existing scorers - baseScorer is required for updates
  if (!baseScorer) {
    throw new ScorerTransformationError('Base scorer is required for updates');
  }

  // For code-based scorers, only sample rate and filter string can be updated
  if (baseScorer.type === 'custom-code' && formData.scorerType === 'custom-code') {
    const updatedScorer: CustomCodeScorer = {
      ...baseScorer,
      sampleRate: formData.sampleRate,
      filterString: formData.filterString || '',
      // Keep all other fields from baseScorer unchanged
    };
    return updatedScorer;
  }

  // For updating llm based scorers
  const updatedScorer: ScheduledScorer = {
    ...baseScorer,
    name: formData.name,
    sampleRate: formData.sampleRate,
    filterString: formData.filterString || '',
  };

  // Update LLM-specific fields if this is an LLM scorer
  if (baseScorer && baseScorer.type === 'llm' && formData.scorerType === 'llm') {
    const llmFormData = formData as LLMScorerFormData;
    (updatedScorer as LLMScorer).llmTemplate = llmFormData.llmTemplate as LLMTemplate;

    // Add guidelines if this is a Guidelines or ConversationalGuidelines scorer
    if (isGuidelinesTemplate(llmFormData.llmTemplate)) {
      (updatedScorer as LLMScorer).guidelines = (llmFormData.guidelines || '')
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean);
    }

    // Add instructions for instructions judges (Custom, Safety, RelevanceToQuery)
    if (llmFormData.isInstructionsJudge) {
      (updatedScorer as LLMScorer).instructions = llmFormData.instructions;
      (updatedScorer as LLMScorer).outputType = formDataToOutputTypeSpec(llmFormData);
    }
    (updatedScorer as LLMScorer).is_instructions_judge = llmFormData.isInstructionsJudge;

    // Add model for all LLM scorers
    (updatedScorer as LLMScorer).model = llmFormData.model || undefined;
  }

  return updatedScorer;
}
