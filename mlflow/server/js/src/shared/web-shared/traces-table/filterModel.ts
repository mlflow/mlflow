/**
 * Generic multi-clause filter AST for the traces filter popover. The shared layer owns only the
 * neutral model + UI-driving helpers; how a clause compiles into a server-side filter string is
 * API-specific and stays with the consumer.
 */

/**
 * Filter operators. Values are the literal comparison tokens most search backends expect, so a
 * consumer's compiler can flow them straight into the emitted clause.
 */
export enum FilterOp {
  EQUALS = '=',
  NOT_EQUALS = '!=',
  GREATER_THAN = '>',
  LESS_THAN = '<',
  GREATER_THAN_OR_EQUALS = '>=',
  LESS_THAN_OR_EQUALS = '<=',
  CONTAINS = 'CONTAINS',
  RLIKE = 'RLIKE',
}

/** How a clause's value input renders: a fixed single-select, a numeric input, or free text. */
export type FilterValueInputKind = 'select' | 'number' | 'text';

/** One selectable option for a `'select'`-kind value input. `label` is already localized. */
export interface FilterFieldSelectOption {
  value: string;
  label: string;
}

/**
 * Declares one filterable field for the popover: its id (what a clause stores), its localized
 * `label`, the operators it offers (first entry is the default), how its value input renders, and —
 * for a `'select'` input — the options. Consumers build the field list; the popover is fully driven
 * by it, so it hardcodes no field names.
 */
export interface FilterFieldDef {
  id: string;
  label: string;
  /** Operators offered for this field. The first entry is the default when the field is selected. */
  operators: FilterOp[];
  valueInput: FilterValueInputKind;
  /** Options for a `'select'` value input; ignored otherwise. */
  options?: FilterFieldSelectOption[];
  /**
   * Localized placeholder for a `'text'`/`'number'` value input, letting a field surface a required
   * unit or format hint (e.g. "Time in milliseconds" for a duration). Consumer-localized like
   * `label`; falls back to a generic "Value" when omitted. Ignored for a `'select'` input.
   */
  valuePlaceholder?: string;
  /**
   * When true, the field needs a free-text key alongside its value (e.g. an arbitrary tag or
   * metadata key). The popover renders a Key input and the clause carries `key`; a clause for such a
   * field is complete only once its key is non-blank.
   */
  requiresKey?: boolean;
  /** Localized placeholder for the Key input; used only when `requiresKey`. */
  keyPlaceholder?: string;
  /**
   * How the Key input renders when `requiresKey`: a plain `'text'` input (the default) or a
   * `'combobox'` that suggests `keyOptions` while still accepting a freeform-typed key. Purely a
   * render choice — the key stays a string either way.
   */
  keyInput?: 'text' | 'combobox';
  /** Suggested key options for a `'combobox'` key input (localized `label`); ignored otherwise. */
  keyOptions?: FilterFieldSelectOption[];
}

/** A single Field + Operator + Value clause. */
export interface FilterClause {
  field: string;
  operator: FilterOp;
  value: string;
  /**
   * Free-text key for a `requiresKey` field (e.g. the tag/metadata name). Present iff the clause's
   * field `requiresKey`; `undefined` otherwise.
   */
  key?: string;
}

/** Ordered list of clauses; the whole set is typically ANDed together at compile time. */
export type TraceFilterModel = FilterClause[];

export const EMPTY_FILTER_MODEL: TraceFilterModel = [];

/**
 * A clause is "complete" (worth compiling / counting) once it has a field, operator, and value — and,
 * for a `requiresKey` field (signalled by a non-`undefined` `key`), a non-blank key too.
 */
export const isClauseComplete = (clause: FilterClause): boolean =>
  Boolean(clause.field) &&
  Boolean(clause.operator) &&
  clause.value.trim().length > 0 &&
  (clause.key === undefined || clause.key.trim().length > 0);

/** Number of active (complete) clauses — shown as a badge on the Filter button. */
export const countActiveFilters = (model: TraceFilterModel): number => model.filter(isClauseComplete).length;

/**
 * A blank clause seeded from the first field, so opening the popover shows one ready-to-fill row.
 * Returns a safe fallback when the field list is empty (the popover renders no field options then).
 */
export const makeEmptyClause = (fields: FilterFieldDef[]): FilterClause => {
  const first = fields[0];
  return {
    field: first?.id ?? '',
    operator: first?.operators[0] ?? FilterOp.EQUALS,
    value: '',
    // Seed `key: ''` for a key-requiring field so the invariant "key defined iff field requiresKey" holds.
    ...(first?.requiresKey ? { key: '' } : {}),
  };
};
