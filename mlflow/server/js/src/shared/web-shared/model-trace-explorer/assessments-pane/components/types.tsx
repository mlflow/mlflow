// common types for all assessment value input fields
export type AssessmentValueInputFieldProps = {
  value: string | number | boolean;
  valueError?: React.ReactNode;
  setValue: (value: string | number | boolean) => void;
  setValueError: (error: string | null) => void;
  isSubmitting: boolean;
};
