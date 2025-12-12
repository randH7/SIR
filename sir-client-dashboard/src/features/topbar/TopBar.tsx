export function TopBar() {
  return (
    <header className="h-16 px-4 flex items-center justify-between border-b border-neutral-900 bg-neutral-950">
      <div className="flex items-center gap-3">
        <div className="h-9 w-9 rounded-xl bg-neutral-800" />
        <div className="leading-tight">
          <div className="text-sm font-semibold">
            SIR — Smart Incident Radar
          </div>
          <div className="text-xs text-neutral-400">
            Live incidents + predicted risk signals
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-xs text-neutral-400">City</span>
        <div className="px-3 py-1.5 rounded-xl bg-neutral-900 border border-neutral-800 text-xs">
          Riyadh (Demo)
        </div>
      </div>
    </header>
  );
}
