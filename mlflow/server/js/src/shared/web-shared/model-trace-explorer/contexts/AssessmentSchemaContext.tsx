import React, { createContext, useContext, type ReactNode, useMemo } from 'react';

import { parseAssessmentSchemas } from './AssessmentSchemaContext.utils';
import type { Assessment } from '../ModelTrace.types';
import type { AssessmentFormInputDataType } from '../assessments-pane/AssessmentsPane.utils';

export type AssessmentSchema = {
  name: string;
  assessmentType: 'feedback' | 'expectation';
  dataType: AssessmentFormInputDataType;
};

interface AssessmentSchemaContextValue {
  schemas: AssessmentSchema[];
}

const AssessmentSchemaContext = createContext<AssessmentSchemaContextValue>({
  schemas: [],
});

interface AssessmentSchemaContextProviderProps {
  children: ReactNode;
  assessments: Assessment[];
}

export const AssessmentSchemaContextProvider: React.FC<AssessmentSchemaContextProviderProps> = ({
  children,
  assessments,
}) => {
  const schemas = useMemo(() => parseAssessmentSchemas(assessments), [assessments]);
  const value: AssessmentSchemaContextValue = useMemo(
    () => ({
      schemas,
    }),
    [schemas],
  );

  return <AssessmentSchemaContext.Provider value={value}>{children}</AssessmentSchemaContext.Provider>;
};

export const useAssessmentSchemas = (): AssessmentSchemaContextValue => {
  return useContext(AssessmentSchemaContext);
};
