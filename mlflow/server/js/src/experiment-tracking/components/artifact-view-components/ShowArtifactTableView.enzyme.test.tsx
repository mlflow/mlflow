/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow, mount } from 'enzyme';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import ShowArtifactTableView from './ShowArtifactTableView';
// @ts-expect-error TS(7016): Could not find a declaration file for module 'papa... Remove this comment to see the full error message
import Papa from 'papaparse';

describe('ShowArtifactTableView', () => {
  let wrapper: any;
  let minimalProps: any;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      path: 'fakePath.csv',
      runUuid: 'fakeUuid',
    };
    // Mock the `getArtifact` function to avoid spurious network errors
    // during testing
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve('some content');
    });
    commonProps = { ...minimalProps, getArtifact };
    wrapper = shallow(<ShowArtifactTableView {...commonProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactTableView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render raw file text if parsing invalid CSV', (done) => {
    const fileContents = 'abcd\n&&&&&';
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(fileContents);
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(<ShowArtifactTableView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.mlflow-ShowArtifactPage').length).toBe(1);
      expect(wrapper.find('.text-area-border-box').length).toBe(1);
      expect(wrapper.find('.text-area-border-box').text()).toBe(fileContents);
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should only render the first 500 rows when the number of rows is larger than 500', (done) => {
    const data = Array(600).fill({ a: 0, b: 1 });
    const fileContents = Papa.unparse(data);

    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(fileContents);
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(<ShowArtifactTableView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(
        wrapper.find('tbody').findWhere((n: any) => n.name() === 'tr' && n.prop('aria-hidden') !== 'true').length,
      ).toBe(500);
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render CSV file correctly', (done) => {
    const data = Array(2).fill({ a: '0', b: '1' });
    const fileContents = Papa.unparse(data);

    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(fileContents);
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(<ShowArtifactTableView {...props} />);
    setImmediate(() => {
      wrapper.update();
      // Handle matching table headers
      const headerTextNodes = wrapper
        .find('thead')
        .find('tr')
        .findWhere((n: any) => n.name() === 'span' && n.text() !== '')
        .children();
      const csvHeaderValues = headerTextNodes.map((c: any) => c.text());
      expect(csvHeaderValues).toEqual(Object.keys(data[0]));

      // Handle matching row values
      const rowTextNodes = wrapper
        .find('tbody')
        .findWhere((n: any) => n.name() === 'tr' && n.prop('aria-hidden') !== 'true')
        .children();
      const csvPreviewValues = rowTextNodes.map((c: any) => c.text());
      const flatData = data.flatMap((d) => [d.a, d.b]);
      expect(csvPreviewValues).toEqual(flatData);
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render TSV file correctly', (done) => {
    const data = Array(2).fill({ a: '0', b: '1' });
    const fileContents = Papa.unparse(data, { delimiter: '\t' });

    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(fileContents);
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(<ShowArtifactTableView {...props} />);
    setImmediate(() => {
      wrapper.update();
      // Handle matching table headers
      const headerTextNodes = wrapper
        .find('thead')
        .find('tr')
        .findWhere((n: any) => n.name() === 'span' && n.text() !== '')
        .children();
      const csvHeaderValues = headerTextNodes.map((c: any) => c.text());
      expect(csvHeaderValues).toEqual(Object.keys(data[0]));

      // Handle matching row values
      const rowTextNodes = wrapper
        .find('tbody')
        .findWhere((n: any) => n.name() === 'tr' && n.prop('aria-hidden') !== 'true')
        .children();
      const csvPreviewValues = rowTextNodes.map((c: any) => c.text());
      const flatData = data.flatMap((d) => [d.a, d.b]);
      expect(csvPreviewValues).toEqual(flatData);
      done();
    });
  });
});
