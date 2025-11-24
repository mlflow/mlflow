import { css, keyframes } from '@emotion/react';
// This is a very simple PRNG that is seeded (so that the output is deterministic).
// We need this in order to produce a random ragged edge for the table skeleton.
export function pseudoRandomNumberGeneratorFromSeed(seed) {
    // This is a simple way to get a consistent number from a string;
    // `charCodeAt` returns a number between 0 and 65535, and we then just add them all up.
    const seedValue = seed
        .split('')
        .map((char) => char.charCodeAt(0))
        .reduce((prev, curr) => prev + curr, 0);
    // This is a simple sine wave function that will always return a number between 0 and 1.
    // This produces a value akin to `Math.random()`, but has deterministic output.
    // Of course, sine curves are not a perfectly random distribution between 0 and 1, but
    // it's close enough for our purposes.
    return Math.sin(seedValue) / 2 + 0.5;
}
// This is a simple Fisher-Yates shuffler using the above PRNG.
export function shuffleArray(arr, seed) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(pseudoRandomNumberGeneratorFromSeed(seed + String(i)) * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}
// Finally, we shuffle a list off offsets to apply to the widths of the cells.
// This ensures that the cells are not all the same width, but that they are
// random to produce a more realistic looking skeleton.
export function getOffsets(seed) {
    return shuffleArray([48, 24, 0], seed);
}
const skeletonLoading = keyframes({
    '0%': {
        backgroundPosition: '100% 50%',
    },
    '100%': {
        backgroundPosition: '0 50%',
    },
});
export const genSkeletonAnimatedColor = (theme, frameRate = 60) => {
    // TODO: Pull this from the themes; it's not currently available.
    const color = theme.isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(31, 38, 45, 0.1)';
    // Light mode value copied from Ant's Skeleton animation
    const colorGradientEnd = theme.isDarkMode ? 'rgba(99, 99, 99, 0.24)' : 'rgba(129, 129, 129, 0.24)';
    return css({
        animationDuration: '1.4s',
        background: `linear-gradient(90deg, ${color} 25%, ${colorGradientEnd} 37%, ${color} 63%)`,
        backgroundSize: '400% 100%',
        animationName: skeletonLoading,
        animationTimingFunction: `steps(${frameRate}, end)`,
        // Based on data from perf dashboard, p95 loading time goes up to about 20s, so about 14 iterations is needed.
        animationIterationCount: 14,
    });
};
//# sourceMappingURL=utils.js.map