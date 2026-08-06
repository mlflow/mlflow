import { describe, expect, it } from '@jest/globals';

import { validateAndPrepareMessages, validateTemplate } from './validateA2uiMessages';

// A concrete (marker-free) per-trace message stream, as `resolveTemplate`
// produces it — the input `validateAndPrepareMessages` strict-validates before
// the stream reaches the (catalog-unaware) renderer.
const resolvedMessages = (components: unknown[]) => [
  {
    version: 'v0.9',
    updateComponents: {
      surfaceId: 'main',
      components,
    },
  },
];

const prepare = (components: unknown[]) =>
  validateAndPrepareMessages(resolvedMessages(components), { surfaceId: 'surface-1', catalogId: 'catalog-1' });

const assessmentBoardTemplate = (childrenMarker: Record<string, unknown>) => [
  {
    version: 'v0.9',
    updateComponents: {
      surfaceId: 'main',
      components: [
        {
          id: 'root',
          component: 'AssessmentBoard',
          title: 'Assessments',
          children: childrenMarker,
        },
      ],
    },
  },
];

const columnTemplate = (children: unknown) => [
  {
    version: 'v0.9',
    updateComponents: {
      surfaceId: 'main',
      components: [
        { id: 'root', component: 'Column', children },
        { id: 'child-a', component: 'Text', text: 'A' },
      ],
    },
  },
];

// TreeView / TreeNode and DataTable were removed from the catalog, so a view
// saved while they existed no longer validates. The closed allowlist rejects them
// on every path, which surfaces the "rebuild it with the assistant" placeholder
// instead of forwarding an unknown component to the renderer.
describe('validateTemplate removed catalog primitives', () => {
  const componentTemplate = (component: Record<string, unknown>) => [
    { version: 'v0.9', updateComponents: { surfaceId: 'main', components: [component] } },
  ];

  it.each(['TreeView', 'TreeNode'])('rejects a stored %s template as an unknown component type', (component) => {
    const result = validateTemplate(componentTemplate({ id: 'root', component, children: ['child-a'] }));
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining(`unknown component type "${component}"`),
    });
  });

  it('rejects a stored DataTable template as an unknown component type', () => {
    const result = validateTemplate(
      componentTemplate({
        id: 'root',
        component: 'DataTable',
        title: 'Tool Performance',
        columns: [{ label: 'Tool' }],
      }),
    );
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining('unknown component type "DataTable"'),
    });
  });

  it.each(['spanTree', 'toolRows'])('rejects the removed "%s" source', (source) => {
    const result = validateTemplate(componentTemplate({ id: 'root', component: 'Text', text: { $source: source } }));
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining(`references unknown $source "${source}"`),
    });
  });
});

describe('validateTemplate children source binding', () => {
  it('accepts literal child ids on a layout container', () => {
    const result = validateTemplate(columnTemplate(['child-a']));
    expect(result).toEqual({ ok: true, messages: expect.any(Array) });
  });

  it('accepts a structural source on AssessmentBoard children', () => {
    const result = validateTemplate(assessmentBoardTemplate({ $source: 'assessments' }));
    expect(result).toEqual({ ok: true, messages: expect.any(Array) });
  });

  it('rejects a scalar source bound to children', () => {
    const result = validateTemplate(columnTemplate({ $source: 'metrics.latency' }));
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining('binds a non-structural "metrics.latency" source to "children"'),
    });
  });

  it('rejects spanField bound to children', () => {
    const result = validateTemplate(columnTemplate({ $source: 'spanField', spanRef: 'root', field: 'outputs' }));
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining('binds a non-structural "spanField" source to "children"'),
    });
  });
});

// The validator is a CLOSED catalog allowlist. It must reject any component type
// or prop the catalog does not define so that no injected component
// (iframe/script/etc.) or unlisted field reaches the Console DOM via the
// catalog-unaware A2UI renderer.
describe('validateAndPrepareMessages catalog allowlist', () => {
  it('accepts valid basic and custom catalog components', () => {
    const result = prepare([
      { id: 'root', component: 'Column', children: ['row', 'txt'], align: 'stretch' },
      { id: 'row', component: 'Row', children: ['stat'], align: 'stretch' },
      { id: 'stat', component: 'StatCard', value: '14', label: 'Tool calls', icon: 'wrench', weight: 1 },
      { id: 'txt', component: 'Text', text: 'Summary', variant: 'h4', weight: 1 },
    ]);
    expect(result).toEqual({ ok: true, messages: expect.any(Array) });
  });

  it('rejects an unknown / injected component type', () => {
    const result = prepare([
      { id: 'root', component: 'Column', children: ['evil'] },
      { id: 'evil', component: 'iframe', src: 'https://evil.example/xss' },
    ]);
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining('unknown component type "iframe"'),
    });
  });

  it('rejects an unlisted prop on a basic component (no unknown-field passthrough)', () => {
    const result = prepare([{ id: 'root', component: 'Text', text: 'hi', onClick: 'doEvil()' }]);
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining('invalid props'),
    });
  });

  it('rejects a raw dangerouslySetInnerHTML-style prop on a basic layout component', () => {
    const result = prepare([
      {
        id: 'root',
        component: 'Row',
        children: ['x'],
        dangerouslySetInnerHTML: { __html: '<img src=x onerror=alert(1)>' },
      },
    ]);
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining('invalid props'),
    });
  });

  it('rejects an unlisted prop on a custom component', () => {
    const result = prepare([
      { id: 'root', component: 'Column', children: ['stat'] },
      { id: 'stat', component: 'StatCard', value: '1', label: 'x', bogusProp: true },
    ]);
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining('invalid props'),
    });
  });

  it('accepts the "justify" prop on Row/Column (a renderer-supported layout prop)', () => {
    const result = prepare([
      { id: 'root', component: 'Column', children: ['row'], justify: 'spaceBetween' },
      { id: 'row', component: 'Row', children: ['txt'], justify: 'center', align: 'stretch' },
      { id: 'txt', component: 'Text', text: 'Hi' },
    ]);
    expect(result).toEqual({ ok: true, messages: expect.any(Array) });
  });

  // The allowlist is keyed on a null-prototype object, so a component named after
  // an Object.prototype member is rejected (not silently accepted) and never
  // crashes the strict per-trace validator via a truthy non-schema lookup.
  it.each(['constructor', 'toString', 'hasOwnProperty', '__proto__', 'valueOf'])(
    'rejects a component whose type collides with Object.prototype member "%s"',
    (name) => {
      const result = prepare([
        { id: 'root', component: 'Column', children: ['evil'] },
        { id: 'evil', component: name },
      ]);
      expect(result).toMatchObject({
        ok: false,
        error: expect.stringContaining(`unknown component type "${name}"`),
      });
    },
  );
});

// `resolveTemplate` substitutes an empty root Column when a `renderIfSpan` prune
// reaches the root, since a rootless stream is rejected wholesale here. That
// fallback is only useful if it survives this strict validation.
describe('validateAndPrepareMessages collapsed root', () => {
  it('accepts a root left with no children', () => {
    const result = prepare([{ id: 'root', component: 'Column', children: [] }]);
    expect(result).toEqual({ ok: true, messages: expect.any(Array) });
  });

  it('rejects a stream whose only root was removed', () => {
    const result = prepare([{ id: 'orphan', component: 'Text', text: 'hi' }]);
    expect(result).toMatchObject({ ok: false, error: expect.stringContaining('no "root" component') });
  });
});

describe('validateTemplate catalog allowlist', () => {
  const templateWith = (components: unknown[]) => [
    { version: 'v0.9', updateComponents: { surfaceId: 'main', components } },
  ];

  it('rejects an unknown / injected component type at save time', () => {
    const result = validateTemplate(
      templateWith([
        { id: 'root', component: 'Column', children: ['evil'] },
        { id: 'evil', component: 'script', text: 'alert(1)' },
      ]),
    );
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining('unknown component type "script"'),
    });
  });

  it('still accepts a basic component whose data prop holds a binding marker', () => {
    const result = validateTemplate(
      templateWith([{ id: 'root', component: 'Text', text: { $source: 'spanField', spanRef: 'root', field: 'name' } }]),
    );
    expect(result).toEqual({ ok: true, messages: expect.any(Array) });
  });

  it('rejects a component whose type collides with an Object.prototype member', () => {
    const result = validateTemplate(
      templateWith([
        { id: 'root', component: 'Column', children: ['evil'] },
        { id: 'evil', component: 'constructor' },
      ]),
    );
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining('unknown component type "constructor"'),
    });
  });

  // A2UI marks `id` optional, so an id-less component clears the envelope schema.
  // The per-trace validator rejects it, so accepting it at save time would persist
  // a view that can never render.
  it.each([
    ['an absent id', { component: 'Text', text: 'orphan' }],
    ['an empty id', { id: '', component: 'Text', text: 'orphan' }],
  ])('rejects a template component with %s', (_label, orphan) => {
    const result = validateTemplate(templateWith([{ id: 'root', component: 'Column', children: ['a'] }, orphan]));
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining('missing a non-empty "id"'),
    });
  });
});

// A2UI types `updateDataModel.value` as `z.any()`, and components can read it
// back through a `{ "path": ... }` DynamicString binding, so the data model
// would otherwise be an unchecked route around the component-level rules.
describe('validateTemplate data model', () => {
  const templateWithDataModel = (value: unknown) => [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [{ id: 'root', component: 'Text', text: { path: '/heading' } }],
      },
    },
    { version: 'v0.9', updateDataModel: { surfaceId: 'main', value } },
  ];

  it('accepts literal data-model values bound by path', () => {
    const result = validateTemplate(templateWithDataModel({ heading: 'Trace summary', counts: [1, 2, 3] }));
    expect(result).toEqual({ ok: true, messages: expect.any(Array) });
  });

  it('rejects a "#span:" deeplink smuggled into the data model', () => {
    const result = validateTemplate(templateWithDataModel({ heading: 'See [the call](#span:span-abc-123)' }));
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining('"#span:" deeplink'),
    });
  });

  it('rejects a "#span:" deeplink nested inside arrays and objects', () => {
    const result = validateTemplate(templateWithDataModel({ rows: [{ label: 'ok' }, { label: '#span:abc' }] }));
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining('"#span:" deeplink'),
    });
  });

  // Markers are never resolved in the data model, so even well-formed ones would
  // render as raw JSON — reject them rather than persist a broken binding.
  it.each([
    ['a valid $source marker', { heading: { $source: 'metrics.latency' } }, '$source'],
    ['an unknown $source marker', { heading: { $source: 'metrics.bogus' } }, '$source'],
    ['a $spanRef marker', { heading: { $spanRef: 'root' } }, '$spanRef'],
  ])('rejects %s in the data model', (_label, value, marker) => {
    const result = validateTemplate(templateWithDataModel(value));
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining(`"${marker}" marker`),
    });
  });
});
