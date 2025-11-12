import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { Tree } from './Tree';
describe('Tree component accessibility', () => {
    it('renders a div with role="tree" and a nested div containing an input', () => {
        render(_jsx(Tree, { treeData: [{ title: 'Node 1', key: '0-0' }], defaultExpandedKeys: ['0-0'] }));
        // Check if there is a div with role="tree"
        const treeDiv = screen.getByRole('tree');
        expect(treeDiv).toBeInTheDocument();
        // Inside the div with role="tree", find the input element
        const inputElement = screen.getByLabelText('for screen reader');
        expect(inputElement).toBeInTheDocument();
        expect(inputElement.tagName.toLowerCase()).toBe('input');
        // Find the parent div of the input
        const parentDiv = inputElement.closest('div');
        expect(parentDiv).toBeInTheDocument();
        // Check if the parent div has the role="treeitem" to satisfy ARIA role hierarchy
        // ADO ID: 3448106 - Accessibility fix for ensuring ARIA child role 'treeitem' under 'tree'.
        // https://databricks.atlassian.net/browse/ES-1252757
        // PR for the patch https://github.com/databricks/universe/pull/713091
        expect(parentDiv).toHaveAttribute('role', 'treeitem');
    });
});
//# sourceMappingURL=Tree.test.js.map