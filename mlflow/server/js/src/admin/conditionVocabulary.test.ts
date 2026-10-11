import { describe, it, expect } from '@jest/globals';

import {
  findUnsupportedConditionIdentifiers,
  getConditionRequestIdentifiers,
  getConditionResourceIdentifiers,
} from './types';

describe('condition vocabulary — what each resource type accepts', () => {
  it('offers alias only on the types that own aliases', () => {
    // A version's alias list names aliases stored on its parent, so a version is not
    // alias-owning (D18) and an alias clause on one is refused by the server.
    expect(getConditionRequestIdentifiers('registered_model')).toContain('alias');
    expect(getConditionRequestIdentifiers('registered_model_version')).not.toContain('alias');
    expect(getConditionResourceIdentifiers('prompt')).toContain('aliases.<name>');
    expect(getConditionResourceIdentifiers('prompt_version')).not.toContain('aliases.<name>');
  });

  it('offers the dotted tag form on both sides and the flat forms only on the request side', () => {
    // The flat names match any tag the request writes, which has no resource-side meaning:
    // a resource simply has the tags it has.
    expect(getConditionRequestIdentifiers('run')).toEqual(['tag_key', 'tag_value', 'tags.<key>']);
    expect(getConditionResourceIdentifiers('run')).toEqual(['tags.<key>']);
  });
});

describe('findUnsupportedConditionIdentifiers', () => {
  it('accepts the vocabulary each side owns', () => {
    expect(findUnsupportedConditionIdentifiers("tag_key != 'lifecycle'", 'run', 'request')).toEqual([]);
    expect(findUnsupportedConditionIdentifiers("tags.team IN ('a','b')", 'run', 'request')).toEqual([]);
    expect(findUnsupportedConditionIdentifiers("tags.lifecycle != 'prod'", 'run', 'resource')).toEqual([]);
    expect(findUnsupportedConditionIdentifiers("alias != 'champion'", 'registered_model', 'request')).toEqual([]);
    expect(findUnsupportedConditionIdentifiers("aliases.champion = 'v1'", 'registered_model', 'resource')).toEqual([]);
  });

  it('rejects an alias clause on a type that owns no aliases', () => {
    // The finding that prompted this: a Run condition naming `alias` could be staged and
    // submitted even though the backend refuses aliases for that type.
    expect(findUnsupportedConditionIdentifiers("alias != 'champion'", 'run', 'request')).toEqual(['alias']);
    expect(findUnsupportedConditionIdentifiers("aliases.champion = 'v1'", 'run', 'resource')).toEqual([
      'aliases.champion',
    ]);
  });

  it('rejects a flat request identifier used in a target condition', () => {
    // Easy to get wrong because the two fields accept overlapping syntax, and a clause
    // copied from one to the other looks right.
    expect(findUnsupportedConditionIdentifiers("tag_key != 'lifecycle'", 'run', 'resource')).toEqual(['tag_key']);
  });

  it('rejects an identifier that is in no vocabulary at all', () => {
    expect(findUnsupportedConditionIdentifiers("stage = 'Production'", 'run', 'request')).toEqual(['stage']);
    expect(findUnsupportedConditionIdentifiers("params.alpha = '1'", 'run', 'resource')).toEqual(['params.alpha']);
  });

  it('does not read a quoted value as a clause', () => {
    // Without blanking literals the value here would scan as an `alias` clause and the
    // admin would be told a valid condition is unsupported.
    expect(findUnsupportedConditionIdentifiers("tags.note = 'alias = champion'", 'run', 'resource')).toEqual([]);
  });

  it('reports each offending identifier once, in the form it was typed', () => {
    const found = findUnsupportedConditionIdentifiers(
      "alias != 'a' AND alias != 'b' AND stage = 'c'",
      'run',
      'request',
    );
    expect(found).toEqual(['alias', 'stage']);
  });

  it('stays silent on a filter it cannot parse, leaving the server as the authority', () => {
    // Deliberately not a second parser: this guards the mistakes the vocabulary makes easy
    // and defers syntax, clause counts and everything else to the backend.
    expect(findUnsupportedConditionIdentifiers('not a filter at all', 'run', 'request')).toEqual([]);
    expect(findUnsupportedConditionIdentifiers('', 'run', 'request')).toEqual([]);
  });
});
