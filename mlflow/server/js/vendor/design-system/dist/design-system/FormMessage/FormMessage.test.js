import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { FormMessage } from './FormMessage';
import { DesignSystemProvider } from '../DesignSystemProvider';
describe('FormMessage', function () {
    function renderComponent({ type = 'warning', message = 'some message', ...rest } = {}) {
        return render(_jsx(DesignSystemProvider, { children: _jsx(FormMessage, { type: type, message: message, ...rest }) }));
    }
    it('renders message', async () => {
        renderComponent();
        expect(screen.getByText('some message')).toBeInTheDocument();
    });
});
//# sourceMappingURL=FormMessage.test.js.map