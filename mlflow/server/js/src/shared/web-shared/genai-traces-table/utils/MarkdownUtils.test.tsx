import { describe, test, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import React from 'react';

import { MarkdownConverterProvider, useMarkdownConverter } from './MarkdownUtils';

// Helper component that renders makeHTML output via dangerouslySetInnerHTML,
// mirroring how the real consumers (EvaluationsReviewTextBox, etc.) use it.
const MakeHTMLConsumer = ({ input }: { input?: string }) => {
  const { makeHTML } = useMarkdownConverter();
  const html = makeHTML(input);
  return html ? (
    // eslint-disable-next-line react/no-danger
    <span data-testid="output" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span data-testid="output" />
  );
};

describe('MarkdownUtils', () => {
  describe('default makeHTML (no provider)', () => {
    test('strips script tags from input', () => {
      render(<MakeHTMLConsumer input='<script>alert("xss")</script>' />);
      const output = screen.getByTestId('output');
      expect(output.innerHTML).not.toContain('<script>');
      expect(output.innerHTML).not.toContain('alert');
    });

    test('strips onerror event handlers from img tags', () => {
      render(<MakeHTMLConsumer input='<img src=x onerror="alert(1)">' />);
      const output = screen.getByTestId('output');
      expect(output.innerHTML).not.toContain('onerror');
    });

    test('strips javascript: protocol from href', () => {
      render(<MakeHTMLConsumer input='<a href="javascript:alert(1)">click</a>' />);
      const output = screen.getByTestId('output');
      // eslint-disable-next-line no-script-url
      expect(output.innerHTML).not.toContain('javascript:');
    });

    test('preserves safe HTML', () => {
      render(<MakeHTMLConsumer input="<strong>bold</strong> and <em>italic</em>" />);
      const output = screen.getByTestId('output');
      expect(output.innerHTML).toContain('<strong>bold</strong>');
      expect(output.innerHTML).toContain('<em>italic</em>');
    });

    test('returns undefined for undefined input', () => {
      render(<MakeHTMLConsumer input={undefined} />);
      const output = screen.getByTestId('output');
      expect(output.innerHTML).toBe('');
    });
  });

  describe('with MarkdownConverterProvider', () => {
    test('uses the provided makeHtml instead of default', () => {
      const customMakeHtml = (md?: string) => `<p>${md ?? ''}</p>`;
      render(
        <MarkdownConverterProvider makeHtml={customMakeHtml}>
          <MakeHTMLConsumer input="hello" />
        </MarkdownConverterProvider>,
      );
      const output = screen.getByTestId('output');
      expect(output.innerHTML).toBe('<p>hello</p>');
    });
  });
});
