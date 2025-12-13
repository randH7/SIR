export function Spinner({ size = 18 }: { size?: number }) {
  return (
    <div
      className="inline-block animate-spin rounded-full border-2 border-neutral-700 border-t-neutral-200"
      style={{ width: size, height: size }}
      aria-label="Loading"
    />
  );
}
