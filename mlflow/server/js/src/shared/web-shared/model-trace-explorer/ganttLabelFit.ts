const pendingLabels = new Map<HTMLElement, { beside: HTMLElement; availableWidth: number }>();

const fitLabels = () => {
  const labels = [...pendingLabels].filter(([inside, { beside }]) => inside.isConnected && beside.isConnected);
  pendingLabels.clear();
  for (const [inside, { beside }] of labels) {
    inside.style.display = 'inline-flex';
    beside.style.display = 'inline-flex';
  }
  const fits = labels.map(
    ([inside, { beside, availableWidth }]) => Math.max(inside.offsetWidth, beside.offsetWidth) < availableWidth,
  );
  labels.forEach(([inside, { beside }], index) => {
    inside.style.display = fits[index] ? 'inline-flex' : 'none';
    beside.style.display = fits[index] ? 'none' : 'inline-flex';
  });
};

// Batch visibility writes and width reads across mounted rows before the next paint.
export const fitGanttLabel = (inside: HTMLElement, beside: HTMLElement, availableWidth: number) => {
  if (!pendingLabels.size) queueMicrotask(fitLabels);
  pendingLabels.set(inside, { beside, availableWidth });
};
