import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, test, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LegacyTooltip } from './LegacyTooltip';
import { Button } from '../Button';
const TestFixture = (_jsx(LegacyTooltip, { title: "Tooltip content", children: _jsx(Button, { componentId: "tooltip-test-id", children: "Hover me" }) }));
const TestFixtureLabel = (_jsx(LegacyTooltip, { title: "Tooltip content", useAsLabel: true, children: _jsx(Button, { componentId: "tooltip-test-id", children: "Hover me" }) }));
describe('Tooltip', () => {
    test('should only add aria-describedby when mouse enters', async () => {
        render(TestFixture);
        const button = screen.getByRole('button');
        expect(button).not.toHaveAttribute('aria-describedby');
        await userEvent.hover(button);
        expect(button).toHaveAttribute('aria-describedby');
        expect(await screen.findByRole('tooltip')).toBeInTheDocument();
    });
    test('should remove aria-describedby when mouse leaves', async () => {
        render(TestFixture);
        const button = screen.getByRole('button');
        await userEvent.hover(button);
        expect(button).toHaveAttribute('aria-describedby');
        await userEvent.unhover(button);
        expect(button).not.toHaveAttribute('aria-describedby');
        expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    });
    test('should only add aria-labelledby when mouse enters and useAsLabel is true', async () => {
        render(TestFixtureLabel);
        const button = screen.getByRole('button');
        expect(button).not.toHaveAttribute('aria-labelledby');
        await userEvent.hover(button);
        expect(button).toHaveAttribute('aria-labelledby');
        expect(await screen.findByRole('tooltip')).toBeInTheDocument();
    });
    test('should remove aria-labelledby when mouse leaves and useAsLabel is true', async () => {
        render(TestFixtureLabel);
        const button = screen.getByRole('button');
        await userEvent.hover(button);
        expect(button).toHaveAttribute('aria-labelledby');
        await userEvent.unhover(button);
        expect(button).not.toHaveAttribute('aria-labelledby');
        expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    });
});
//# sourceMappingURL=LegacyTooltip.test.js.map