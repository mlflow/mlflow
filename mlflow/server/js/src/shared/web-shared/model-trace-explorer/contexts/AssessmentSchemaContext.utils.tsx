import { isNil } from 'lodash';

import type { AssessmentSchema } from './AssessmentSchemaContext';
import type { Assessment } from '../ModelTrace.types';
import type { AssessmentFormInputDataType } from '../assessments-pane/AssessmentsPane.utils';
import { getAssessmentValue } from '../assessments-pane/utils';

// this function accepts a flat list of assessments (e.g. the result
// of tracesInfos.flatMap(info => info.assessments)), and returns a
// list of assessment schemas
export const parseAssessmentSchemas = (assessments: Assessment[]): AssessmentSchema[] => {
  // stores all schemas for which we can determine the data type
  const schemaMap: { [assessmentName: string]: AssessmentSchema } = {};
  // stores all schemas with null / undefined values.
  // after parsing all values, we will merge the two,
  // keeping the data type from the schemaMap if it exists,
  // and using `boolean` if it does not.
  const nullsSchemaMap: { [assessmentName: string]: AssessmentSchema } = {};

  for (const assessment of assessments) {
    if (schemaMap[assessment.assessment_name]) {
      continue;
    }

    // NOTE: the getAssessmentValue function does not parse
    // serialized JSON, and just returns them as strings.
    const value = getAssessmentValue(assessment);

    if (isNil(value)) {
      nullsSchemaMap[assessment.assessment_name] = {
        name: assessment.assessment_name,
        assessmentType: 'feedback' in assessment ? 'feedback' : 'expectation',
        dataType: 'boolean',
      };
      continue;
    }

    const isSerializedExpectation = 'expectation' in assessment && 'serialized_value' in assessment.expectation;

    let dataType: AssessmentFormInputDataType;
    switch (typeof value) {
      case 'string':
        dataType = isSerializedExpectation ? 'json' : 'string';
        break;
      case 'boolean':
        dataType = 'boolean';
        break;
      case 'number':
        dataType = 'number';
        break;
      // for unexpected types, just default to boolean
      default:
        dataType = 'boolean';
        break;
    }

    schemaMap[assessment.assessment_name] = {
      name: assessment.assessment_name,
      assessmentType: 'feedback' in assessment ? 'feedback' : 'expectation',
      dataType,
    };
  }

  // combine the two maps, keeping the data type from the schemaMap if it exists,
  for (const [assessmentName, schema] of Object.entries(nullsSchemaMap)) {
    if (!(assessmentName in schemaMap)) {
      schemaMap[assessmentName] = schema;
    }
  }

  return Object.values(schemaMap);
};
