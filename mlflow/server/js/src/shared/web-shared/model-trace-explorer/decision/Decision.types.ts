export interface DecisionQuestionViewModel {
  id: string;
  declaredType?: string;
  instructions?: unknown;
  criteria?: unknown;
  rawQuestion: unknown;
}

export type DecisionInputField =
  | {
      kind: 'field';
      fieldKey: string;
    }
  | {
      kind: 'questions';
      fieldKey: string;
      items: readonly DecisionQuestionViewModel[];
    };

interface DecisionAnswerBase {
  id: string;
  rawAnswer: unknown;
}

export interface DecisionNoulAnswerViewModel extends DecisionAnswerBase {
  kind: 'noul';
  probabilityTrue: number;
}

export interface DecisionChoiceProbability {
  label: string;
  probability: number;
}

export interface DecisionChoiceAnswerViewModel extends DecisionAnswerBase {
  kind: 'choice';
  choice: string;
  confidence: number;
  probabilities: DecisionChoiceProbability[];
}

export interface DecisionScoreLevel {
  score: number;
  description?: unknown;
  probability?: number;
}

export interface DecisionScoreAnswerViewModel extends DecisionAnswerBase {
  kind: 'score';
  score: number;
  confidence: number;
  levels: DecisionScoreLevel[];
  range: {
    min: number;
    max: number;
  };
}

export interface DecisionUnknownAnswerViewModel extends DecisionAnswerBase {
  kind: 'unknown';
  declaredType?: string;
  reason: 'malformed' | 'unsupported';
}

export type DecisionAnswerViewModel =
  | DecisionNoulAnswerViewModel
  | DecisionChoiceAnswerViewModel
  | DecisionScoreAnswerViewModel
  | DecisionUnknownAnswerViewModel;

export interface DecisionViewModel {
  inputs: {
    fields: readonly DecisionInputField[];
  };
  answers: DecisionAnswerViewModel[];
}

export interface DecisionSpanLike {
  chatMessageFormat?: unknown;
  inputs?: unknown;
  outputs?: unknown;
}

export interface DecisionTranslator {
  matches: (span: DecisionSpanLike) => boolean;
  translate: (span: DecisionSpanLike) => DecisionViewModel | null;
}
