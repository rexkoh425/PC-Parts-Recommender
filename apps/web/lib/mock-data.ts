// Hand-written fixtures for the first screens.
// Replaced by lib/demo-api.ts, which serves the controlled demo contract.

export const mockProducts = [
  { id: 'cpu-1', category: 'cpu', brand: 'AMD', model: 'Ryzen 5 7600' },
  { id: 'gpu-1', category: 'gpu', brand: 'NVIDIA', model: 'RTX 4060' },
];

export const mockBuild = {
  id: 'build-1',
  total: 1499,
  items: mockProducts,
};
