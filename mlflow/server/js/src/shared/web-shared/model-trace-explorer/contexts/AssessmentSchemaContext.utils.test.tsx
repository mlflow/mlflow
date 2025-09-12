import { parseAssessmentSchemas } from './AssessmentSchemaContext.utils';
import type { Assessment } from '../ModelTrace.types';
import { MOCK_ASSESSMENT, MOCK_EXPECTATION } from '../ModelTraceExplorer.test-utils';

describe('parseAssessmentSchemas', () => {
  it('should handle empty assessments array', () => {
    const result = parseAssessmentSchemas([]);
    expect(result).toEqual([]);
  });

  it('should parse multiple assessments with different data types', () => {
    const assessments: Assessment[] = [
      MOCK_ASSESSMENT, // string type
      {
        ...MOCK_ASSESSMENT,
        assessment_name: 'numeric',
        feedback: { value: 3 },
      },
      {
        ...MOCK_ASSESSMENT,
        assessment_name: 'undefined',
        feedback: { value: undefined },
      },
      MOCK_EXPECTATION, // json
    ];
    const result = parseAssessmentSchemas(assessments);

    expect(result).toHaveLength(4);
    expect(result).toEqual([
      {
        name: 'Relevance',
        assessmentType: 'feedback',
        dataType: 'string',
      },
      {
        name: 'numeric',
        assessmentType: 'feedback',
        dataType: 'number',
      },
      {
        name: 'expected_facts',
        assessmentType: 'expectation',
        dataType: 'json',
      },
      // nullish assessments get put to the back of the array due
      // to implementation, but since it's a typeahead it shouldn't
      // matter very much
      {
        name: 'undefined',
        assessmentType: 'feedback',
        // undefined should default to boolean dataType
        dataType: 'boolean',
      },
    ]);
  });

  it('should deduplicate assessments with the same assessment_name', () => {
    const assessments: Assessment[] = [
      MOCK_ASSESSMENT,
      { ...MOCK_ASSESSMENT, feedback: { value: 3 } },
      MOCK_EXPECTATION,
    ];
    const result = parseAssessmentSchemas(assessments);

    expect(result).toHaveLength(2);
    expect(result).toEqual([
      {
        name: 'Relevance',
        assessmentType: 'feedback',
        dataType: 'string',
      },
      {
        name: 'expected_facts',
        assessmentType: 'expectation',
        dataType: 'json',
      },
    ]);
  });

  it('should default dataType to first non-null feedback value', () => {
    const assessments: Assessment[] = [
      { ...MOCK_ASSESSMENT, feedback: { value: null } },
      { ...MOCK_ASSESSMENT, feedback: { value: undefined } },
      { ...MOCK_ASSESSMENT, feedback: { value: 3 } },
    ];
    const result = parseAssessmentSchemas(assessments);

    expect(result).toHaveLength(1);
    expect(result).toEqual([
      {
        name: 'Relevance',
        assessmentType: 'feedback',
        dataType: 'number',
      },
    ]);
  });
});
