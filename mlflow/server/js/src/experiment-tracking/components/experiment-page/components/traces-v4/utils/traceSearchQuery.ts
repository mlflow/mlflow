import { FilterOp, type FilterClause, type FilterFieldDef } from '@databricks/web-shared/traces-table';
import {
  formatTraceSearchFieldName,
  formatTraceSearchValue,
  getTraceSearchFieldAliases,
  isSafeTraceSearchKey,
  type TraceSearchField,
} from './traceSearchSuggestions';

export type TraceSearchQueryStatus = 'text' | 'partial' | 'complete' | 'invalid';
export type TraceSearchQueryStage = 'field' | 'key' | 'operator' | 'value' | 'complete';

export interface ParsedTraceSearchQuery {
  status: TraceSearchQueryStatus;
  stage: TraceSearchQueryStage;
  /** True when the input is using a known field or explicit query operator. */
  recognized: boolean;
  /** True when the input should be handled as structured syntax, including invalid syntax. */
  structured: boolean;
  /** An exact field alias with no key or operator can also be submitted as literal text. */
  bareField?: boolean;
  field?: FilterFieldDef;
  searchField?: TraceSearchField;
  key?: string;
  operator?: FilterOp;
  value?: string;
  /** The current key/value/operator prefix used to filter completions. */
  prefix?: string;
  clause?: FilterClause;
}

export type TraceSearchQuerySuggestionKind = 'key' | 'operator' | 'value' | 'apply';

export interface TraceSearchQuerySuggestion {
  id: string;
  kind: TraceSearchQuerySuggestionKind;
  label: string;
  /** The query text to put back into the editable search input. */
  query: string;
  /** Present for a complete value suggestion that can be applied immediately. */
  clause?: FilterClause;
}

interface QueryToken {
  value: string;
  quoted: boolean;
  startsWithQuote: boolean;
  unterminatedQuote: boolean;
}

interface TokenizedQuery {
  tokens: QueryToken[];
  malformed: boolean;
}

interface FieldCandidate {
  field: FilterFieldDef;
  aliases: string[];
}

interface FieldMatch {
  candidate: FieldCandidate;
  tokenCount: number;
  dottedKey?: string;
}

interface OperatorMatch {
  operator?: FilterOp;
  consumed: number;
  prefix?: string;
  partial?: boolean;
  invalid?: boolean;
}

const SYMBOL_OPERATORS = new Map<string, FilterOp>([
  [FilterOp.EQUALS, FilterOp.EQUALS],
  [FilterOp.NOT_EQUALS, FilterOp.NOT_EQUALS],
  [FilterOp.GREATER_THAN, FilterOp.GREATER_THAN],
  [FilterOp.LESS_THAN, FilterOp.LESS_THAN],
  [FilterOp.GREATER_THAN_OR_EQUALS, FilterOp.GREATER_THAN_OR_EQUALS],
  [FilterOp.LESS_THAN_OR_EQUALS, FilterOp.LESS_THAN_OR_EQUALS],
]);

const WORD_OPERATORS = new Map<string, FilterOp>([
  [FilterOp.CONTAINS, FilterOp.CONTAINS],
  [FilterOp.RLIKE, FilterOp.RLIKE],
]);

const QUERY_OPERATOR_WORDS = new Set(['contains', 'rlike']);
const UNSUPPORTED_SEARCH_FIELD_IDS = new Set(['cost']);

const compareText = (left: string, right: string): number => (left < right ? -1 : left > right ? 1 : 0);

const tokenize = (input: string): TokenizedQuery => {
  const tokens: QueryToken[] = [];
  let current = '';
  let tokenStarted = false;
  let quoted = false;
  let startsWithQuote = false;
  let unterminatedQuote = false;
  let quote: string | undefined;
  let malformed = false;

  const flush = () => {
    if (tokenStarted) {
      tokens.push({ value: current, quoted, startsWithQuote, unterminatedQuote });
      current = '';
      tokenStarted = false;
      quoted = false;
      startsWithQuote = false;
      unterminatedQuote = false;
    }
  };

  for (let index = 0; index < input.length; index += 1) {
    const character = input[index];

    if (quote) {
      if (character === '\\' && (input[index + 1] === quote || input[index + 1] === '\\')) {
        current += input[index + 1];
        index += 1;
      } else if (character === quote) {
        quote = undefined;
        quoted = true;
      } else {
        current += character;
      }
      continue;
    }

    if (character === '"' || character === "'" || character === '`') {
      if (!tokenStarted) {
        startsWithQuote = true;
      }
      tokenStarted = true;
      quote = character;
      continue;
    }

    if (/\s/.test(character)) {
      flush();
      continue;
    }

    const twoCharacterOperator = `${character}${input[index + 1] ?? ''}`;
    if (
      twoCharacterOperator === FilterOp.NOT_EQUALS ||
      twoCharacterOperator === FilterOp.GREATER_THAN_OR_EQUALS ||
      twoCharacterOperator === FilterOp.LESS_THAN_OR_EQUALS
    ) {
      flush();
      tokens.push({ value: twoCharacterOperator, quoted: false, startsWithQuote: false, unterminatedQuote: false });
      index += 1;
      continue;
    }

    if (character === '=' || character === '>' || character === '<') {
      flush();
      tokens.push({ value: character, quoted: false, startsWithQuote: false, unterminatedQuote: false });
      continue;
    }

    tokenStarted = true;
    current += character;
  }

  if (quote) {
    malformed = true;
    unterminatedQuote = true;
  }
  flush();
  return { tokens, malformed };
};

const makeFieldCandidates = (fields: FilterFieldDef[]): FieldCandidate[] =>
  fields
    .filter((field) => !UNSUPPORTED_SEARCH_FIELD_IDS.has(field.id))
    .map((field) => ({
      field,
      aliases: [formatTraceSearchFieldName(field.id), field.label, ...getTraceSearchFieldAliases(field)]
        .map((alias) => alias.trim().toLocaleLowerCase())
        .filter((alias, index, all) => alias !== '' && alias !== 'assessment' && all.indexOf(alias) === index),
    }));

const splitAlias = (alias: string): string[] => alias.split(/\s+/).filter(Boolean);

const findSimpleField = (candidates: FieldCandidate[], name: string): FieldCandidate | undefined =>
  candidates.find((candidate) => candidate.aliases.some((alias) => !alias.includes(' ') && alias === name));

const matchField = (tokens: QueryToken[], candidates: FieldCandidate[]): FieldMatch | undefined => {
  const first = tokens[0];
  if (!first) return undefined;

  // Dotted syntax is deliberately split only at the first dot so assessment names and tag keys
  // may themselves contain dots.
  if (!first.startsWithQuote) {
    const dot = first.value.indexOf('.');
    if (dot > 0) {
      const candidate = findSimpleField(candidates, first.value.slice(0, dot).toLocaleLowerCase());
      if (candidate) {
        return { candidate, tokenCount: 1, dottedKey: first.value.slice(dot + 1) };
      }
    }
  }

  let best: FieldMatch | undefined;
  for (const candidate of candidates) {
    for (const alias of candidate.aliases) {
      const aliasTokens = splitAlias(alias);
      if (
        aliasTokens.length <= tokens.length &&
        aliasTokens.every((aliasToken, index) => tokens[index].value.toLocaleLowerCase() === aliasToken)
      ) {
        if (!best || aliasTokens.length > best.tokenCount) {
          best = { candidate, tokenCount: aliasTokens.length };
        }
      }
    }
  }
  return best;
};

const isExplicitOperator = (token: QueryToken | undefined): boolean => {
  if (!token || token.quoted) return false;
  return SYMBOL_OPERATORS.has(token.value) || QUERY_OPERATOR_WORDS.has(token.value.toLocaleLowerCase());
};

const parseOperator = (tokens: QueryToken[], index: number): OperatorMatch => {
  const token = tokens[index];
  if (!token) return { consumed: 0 };

  const symbolOperator = SYMBOL_OPERATORS.get(token.value);
  if (symbolOperator) return { operator: symbolOperator, consumed: 1 };

  if (token.value === '!') return { consumed: 1, prefix: '!', partial: true };

  const word = token.value.toLocaleUpperCase();
  const wordOperator = WORD_OPERATORS.get(word);
  if (wordOperator) return { operator: wordOperator, consumed: 1 };

  const isOperatorPrefix = [...WORD_OPERATORS.keys()].some((operator) => operator.startsWith(word));
  return isOperatorPrefix
    ? { consumed: 1, prefix: token.value, partial: true }
    : { consumed: 1, prefix: token.value, invalid: true };
};

const isSafeQueryValue = (value: string): boolean => !value.includes('\0') && !/[\r\n]/.test(value);

const normalizeDuration = (value: string): string | undefined => {
  const match = /^(\d+(?:\.\d+)?)\s*(ms|s)?$/i.exec(value.trim());
  if (!match) return undefined;
  const milliseconds = Number(match[1]) * (match[2]?.toLocaleLowerCase() === 's' ? 1000 : 1);
  return Number.isSafeInteger(milliseconds) ? String(milliseconds) : undefined;
};

const findSearchField = (
  field: FilterFieldDef,
  key: string | undefined,
  searchFields: TraceSearchField[],
): TraceSearchField | undefined => {
  const exact = searchFields.find((searchField) => searchField.field.id === field.id && searchField.key === key);
  if (exact) return exact;
  const anyField = searchFields.find((searchField) => searchField.field.id === field.id);
  if (!key && anyField) return anyField;
  if (!key) return undefined;
  return {
    id: `${field.id}:${key}`,
    field: anyField?.field ?? field,
    key,
    label: formatTraceSearchFieldName(field.id, key),
    values: [],
    aliases: getTraceSearchFieldAliases(field),
  };
};

const makeResult = (
  status: TraceSearchQueryStatus,
  stage: TraceSearchQueryStage,
  recognized: boolean,
  structured: boolean,
  details: Omit<ParsedTraceSearchQuery, 'status' | 'stage' | 'recognized' | 'structured'> = {},
): ParsedTraceSearchQuery => ({ status, stage, recognized, structured, ...details });

/** Parse one editable Field[.Key] Operator Value query into the existing filter AST. */
export const parseTraceSearchQuery = (
  input: string,
  fields: FilterFieldDef[],
  searchFields: TraceSearchField[],
): ParsedTraceSearchQuery => {
  const trimmed = input.trim();
  if (!trimmed) return makeResult('text', 'field', false, false);

  const tokenized = tokenize(trimmed);
  const { tokens } = tokenized;
  const candidates = makeFieldCandidates(fields);
  const fieldMatch = matchField(tokens, candidates);
  const explicitOperator = tokens.some(isExplicitOperator);

  if (!fieldMatch) {
    return makeResult('invalid', 'field', explicitOperator, explicitOperator, {
      prefix: tokens[0]?.value,
    });
  }

  // `buildTraceSearchFields` narrows duration to the operators the V4 backend can safely
  // serialize. Use that effective definition for typed queries too, including when no rows are
  // loaded, so latency `=`/`!=` cannot sneak back in through the broader popover definition.
  const field =
    searchFields.find((searchField) => searchField.field.id === fieldMatch.candidate.field.id)?.field ??
    fieldMatch.candidate.field;
  // Keep an exact field as a query prefix for suggestions, but allow explicit text submission.
  const bareField = tokens.length === fieldMatch.tokenCount && fieldMatch.dottedKey === undefined;

  let cursor = fieldMatch.tokenCount;
  let key = fieldMatch.dottedKey;
  let keyToken: QueryToken | undefined;

  if (field.requiresKey) {
    if (key === undefined) {
      if (!tokens[cursor] || isExplicitOperator(tokens[cursor])) {
        return makeResult('partial', 'key', true, true, {
          field,
          bareField,
          prefix: '',
          searchField: findSearchField(field, undefined, searchFields),
        });
      }
      keyToken = tokens[cursor];
      key = keyToken.value;
      cursor += 1;
    }

    if (key === '') {
      return makeResult('partial', 'key', true, true, {
        field,
        prefix: '',
        searchField: findSearchField(field, undefined, searchFields),
      });
    }
    if (keyToken?.unterminatedQuote || (fieldMatch.dottedKey !== undefined && tokens[0]?.unterminatedQuote)) {
      return makeResult('partial', 'key', true, true, {
        field,
        key,
        prefix: key,
        searchField: findSearchField(field, key, searchFields),
      });
    }
    if (!isSafeTraceSearchKey(field.id, key)) {
      return makeResult('invalid', 'key', true, true, { field, key, prefix: key });
    }
  } else if (fieldMatch.dottedKey !== undefined) {
    return makeResult('invalid', 'field', true, true, { field, prefix: fieldMatch.dottedKey });
  }

  const searchField = findSearchField(field, key, searchFields);
  if (cursor >= tokens.length) {
    return makeResult('partial', 'operator', true, true, {
      field,
      bareField,
      searchField,
      key,
      prefix: '',
    });
  }

  const operatorMatch = parseOperator(tokens, cursor);
  if (operatorMatch.partial) {
    return makeResult('partial', 'operator', true, true, {
      field,
      searchField,
      key,
      prefix: operatorMatch.prefix,
    });
  }
  if (operatorMatch.invalid || !operatorMatch.operator) {
    return makeResult('invalid', 'operator', true, true, {
      field,
      searchField,
      key,
      prefix: operatorMatch.prefix,
    });
  }

  const operator = operatorMatch.operator;
  if (!field.operators.includes(operator)) {
    return makeResult('invalid', 'operator', true, true, { field, searchField, key, operator });
  }
  cursor += operatorMatch.consumed;

  const valueToken = tokens[cursor];
  if (!valueToken) {
    return makeResult('partial', 'value', true, true, {
      field,
      searchField,
      key,
      operator,
      prefix: '',
    });
  }

  let rawValue = valueToken.value;
  let valueTokenCount = 1;
  if (field.id === 'duration' && tokens[cursor + 1] && /^(ms|s)$/i.test(tokens[cursor + 1].value)) {
    rawValue = `${rawValue} ${tokens[cursor + 1].value}`;
    valueTokenCount = 2;
  }
  if (valueToken.unterminatedQuote) {
    return makeResult('partial', 'value', true, true, {
      field,
      searchField,
      key,
      operator,
      value: rawValue,
      prefix: rawValue,
    });
  }
  if (tokens.length > cursor + valueTokenCount || !isSafeQueryValue(rawValue) || rawValue.trim() === '') {
    return makeResult('invalid', 'value', true, true, {
      field,
      searchField,
      key,
      operator,
      value: rawValue,
      prefix: rawValue,
    });
  }

  let value = rawValue;
  if (field.id === 'duration') {
    value = normalizeDuration(rawValue) ?? '';
    if (!value) {
      return makeResult('invalid', 'value', true, true, { field, searchField, key, operator, prefix: rawValue });
    }
  }
  if (field.valueInput === 'select') {
    const option = field.options?.find(
      (candidate) => candidate.value.toLocaleLowerCase() === value.toLocaleLowerCase(),
    );
    if (!option) {
      return makeResult('invalid', 'value', true, true, { field, searchField, key, operator, prefix: rawValue });
    }
    value = option.value;
  }

  const clause: FilterClause = { field: field.id, key, operator, value };
  return makeResult('complete', 'complete', true, true, { field, searchField, key, operator, value, clause });
};

const makeQueryPrefix = (field: FilterFieldDef, key: string | undefined): string =>
  formatTraceSearchFieldName(field.id, key);

const displayFieldLabel = (field: FilterFieldDef, key: string | undefined): string =>
  formatTraceSearchFieldName(field.id, key);

const keySuggestions = (
  field: FilterFieldDef,
  keyPrefix: string,
  searchFields: TraceSearchField[],
): TraceSearchQuerySuggestion[] => {
  const normalizedPrefix = keyPrefix.toLocaleLowerCase();
  const exactKey = searchFields.some(
    (searchField) => searchField.field.id === field.id && searchField.key?.toLocaleLowerCase() === normalizedPrefix,
  );
  if (keyPrefix && exactKey) return [];
  return searchFields
    .filter(
      (searchField) =>
        searchField.field.id === field.id &&
        searchField.key !== undefined &&
        isSafeTraceSearchKey(field.id, searchField.key) &&
        (!normalizedPrefix || searchField.key.toLocaleLowerCase().includes(normalizedPrefix)),
    )
    .sort((left, right) => compareText(left.key ?? '', right.key ?? ''))
    .map((searchField) => ({
      id: `query-key:${field.id}:${searchField.key}`,
      kind: 'key' as const,
      label: searchField.label,
      query: makeQueryPrefix(field, searchField.key),
    }));
};

const valueOptions = (
  field: FilterFieldDef,
  searchField: TraceSearchField | undefined,
): { value: string; label: string }[] =>
  field.valueInput === 'select' ? (field.options ?? []) : (searchField?.values ?? []);

const formatValueLabel = (field: FilterFieldDef, value: string, label?: string): string =>
  formatTraceSearchValue(field, value, label);

/** Return progressive field/key/operator/value completions for a recognized query. */
export const getTraceSearchQuerySuggestions = (
  input: string,
  fields: FilterFieldDef[],
  searchFields: TraceSearchField[],
  maxSuggestions = 5,
): TraceSearchQuerySuggestion[] => {
  const parsed = parseTraceSearchQuery(input, fields, searchFields);
  if (!parsed.recognized || !parsed.structured || !parsed.field) return [];

  const suggestions: TraceSearchQuerySuggestion[] = [];
  const field = parsed.field;
  const displayLabel = displayFieldLabel(field, parsed.key);
  const queryPrefix = makeQueryPrefix(field, parsed.key);

  if (
    field.requiresKey &&
    parsed.status !== 'invalid' &&
    (parsed.stage === 'key' || (parsed.stage === 'operator' && !parsed.prefix))
  ) {
    if (
      parsed.bareField &&
      searchFields.some((searchField) => searchField.field.id === field.id && searchField.key === undefined)
    ) {
      const namespace = `${formatTraceSearchFieldName(field.id)}.`;
      suggestions.push({
        id: `query-key:${field.id}:namespace`,
        kind: 'key',
        label: namespace,
        query: namespace,
      });
    }
    suggestions.push(...keySuggestions(field, parsed.key ?? '', searchFields));
  }

  if (
    parsed.stage === 'operator' &&
    (parsed.status === 'partial' || parsed.status === 'invalid') &&
    (!field.requiresKey || parsed.key !== undefined)
  ) {
    // Once an unsupported/unknown operator is detected, show the field's valid operators as the
    // correction. Never echo the unsupported token back as a suggestion.
    const operatorPrefix = parsed.status === 'partial' ? (parsed.prefix?.toLocaleLowerCase() ?? '') : '';
    suggestions.push(
      ...field.operators
        .filter((operator) => operator.toLocaleLowerCase().includes(operatorPrefix))
        .map((operator) => ({
          id: `query-operator:${field.id}:${parsed.key ?? ''}:${operator}`,
          kind: 'operator' as const,
          label: `${displayLabel} ${operator}`,
          query: `${queryPrefix} ${operator} `,
        })),
    );
  }

  if (parsed.operator && (parsed.stage === 'value' || parsed.status === 'complete')) {
    const operator = parsed.operator;
    const prefix = (parsed.value ?? parsed.prefix ?? '').toLocaleLowerCase();
    const candidates = valueOptions(field, parsed.searchField)
      .filter((candidate) => candidate.label.toLocaleLowerCase().includes(prefix))
      .map((candidate) => ({
        id: `query-value:${field.id}:${parsed.key ?? ''}:${operator}:${candidate.value}`,
        kind: 'value' as const,
        label: `${displayLabel} ${operator} ${formatValueLabel(field, candidate.value, candidate.label)}`,
        query: `${queryPrefix} ${operator} ${formatTraceSearchValue(field, candidate.value)}`,
        clause: { field: field.id, key: parsed.key, operator, value: candidate.value },
      }));
    suggestions.push(...candidates);
  }

  if (parsed.status === 'complete' && parsed.clause) {
    const hasMatchingValue = suggestions.some((suggestion) => suggestion.clause?.value === parsed.clause?.value);
    if (!hasMatchingValue) {
      suggestions.unshift({
        id: 'query-apply',
        kind: 'apply',
        label: `${displayLabel} ${parsed.operator} ${formatValueLabel(field, parsed.value ?? '')}`,
        query: input.trim(),
        clause: parsed.clause,
      });
    }
  }

  return suggestions.slice(0, maxSuggestions);
};
