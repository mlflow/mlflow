import { describe, expect, it } from '@jest/globals';

import { parseCustomView, serializeCustomView, type CustomView } from './customViewDefinition';

// A minimal, well-formed stored view: one updateComponents message with a `root`
// component and no forbidden trace-specific narrative.
const validStored = () => ({
  id: 'view-1',
  name: 'My view',
  label: 'My view',
  instruction: 'show text',
  createdAtMs: 123,
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [{ id: 'root', component: 'Text', text: 'Hello' }],
      },
    },
  ],
});

describe('parseCustomView', () => {
  it('returns undefined for non-objects and shape mismatches', () => {
    expect(parseCustomView(null)).toBeUndefined();
    expect(parseCustomView('nope')).toBeUndefined();
    expect(parseCustomView({ id: 5, template: [] })).toBeUndefined();
    expect(parseCustomView({ id: 'x', template: 'not-an-array' })).toBeUndefined();
  });

  it('parses a well-formed stored view, preserving its template verbatim', () => {
    const stored = validStored();
    const parsed = parseCustomView(stored);
    expect(parsed).toMatchObject({
      id: 'view-1',
      name: 'My view',
      label: 'My view',
      instruction: 'show text',
      createdAtMs: 123,
    });
    // parseCustomView is a shape narrower only: the template is kept verbatim (no
    // validation / normalization at load) and a valid view is never `unreadable`.
    expect(parsed?.template).toEqual(stored.template);
    expect(parsed?.unreadable).toBeUndefined();
  });

  // Template validation is deferred to the selection/render path (so tab load
  // doesn't validate every saved view). parseCustomView therefore keeps ANY
  // structurally-valid view with its ORIGINAL template preserved verbatim and
  // does NOT flag `unreadable` — that Case-2 flag is derived when the view becomes
  // active (see CustomViewDefinitionContext + the render-time gate).
  it('keeps a view with a tampered "#span:" template verbatim without flagging unreadable', () => {
    const tamperedTemplate = [
      {
        version: 'v0.9',
        updateComponents: {
          surfaceId: 'main',
          components: [{ id: 'root', component: 'Text', text: 'jump to #span:abc123' }],
        },
      },
    ];
    const parsed = parseCustomView({ ...validStored(), template: tamperedTemplate });
    expect(parsed?.id).toBe('view-1');
    expect(parsed?.template).toEqual(tamperedTemplate);
    expect(parsed?.unreadable).toBeUndefined();
  });

  it('keeps a view whose template has no root component verbatim without flagging unreadable', () => {
    const noRootTemplate = [
      {
        version: 'v0.9',
        updateComponents: {
          surfaceId: 'main',
          components: [{ id: 'a', component: 'Text', text: 'x' }],
        },
      },
    ];
    const parsed = parseCustomView({ ...validStored(), template: noRootTemplate });
    expect(parsed?.template).toEqual(noRootTemplate);
    expect(parsed?.unreadable).toBeUndefined();
  });

  it('keeps a view that binds a non-structural source to children verbatim without flagging unreadable', () => {
    const badBindingTemplate = [
      {
        version: 'v0.9',
        updateComponents: {
          surfaceId: 'main',
          components: [{ id: 'root', component: 'Column', children: { $source: 'metrics.latency' } }],
        },
      },
    ];
    const parsed = parseCustomView({ ...validStored(), template: badBindingTemplate });
    expect(parsed?.template).toEqual(badBindingTemplate);
    expect(parsed?.unreadable).toBeUndefined();
  });

  it('is idempotent: re-parsing a previously parsed view yields an equal view', () => {
    const first = parseCustomView(validStored()) as CustomView;
    const roundTripped = parseCustomView(JSON.parse(serializeCustomView(first)));
    expect(roundTripped).toEqual(first);
  });

  it('coerces a non-string name/label to a string so the renderer never gets a non-primitive child', () => {
    // Untrusted tag JSON: a non-string `name` with no `label` must not leak
    // through the label fallback (it is rendered directly as a React child).
    const parsed = parseCustomView({
      ...validStored(),
      name: { malicious: 'object' },
      label: undefined,
    });
    expect(parsed?.name).toBe('');
    expect(parsed?.label).toBe('');
    expect(typeof parsed?.label).toBe('string');
  });
});
