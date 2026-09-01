import { useState } from 'react'
import {
  DELIVERABLES,
  STATUS_COLOR,
  STATUS_LABEL,
  deliveryTally,
  type Deliverable,
} from '../lib/deliverables'
import { LANE_COLOR, alpha } from '../lib/theme'

const SECTION_COLOR: Record<Deliverable['section'], string> = {
  Train: LANE_COLOR.classical,
  Predict: LANE_COLOR.quantum,
  Compare: LANE_COLOR.shared,
  Platform: LANE_COLOR.shared,
}

type Filter = 'all' | Deliverable['section']

const FILTERS: Filter[] = ['all', 'Train', 'Predict', 'Compare', 'Platform']

/**
 * Maps every expected deliverable in the problem statement to the section and
 * module that satisfies it. The status column is the honest part: it separates
 * what runs from what is still a mock, so the table cannot be read as claiming
 * a working model where none exists.
 */
export function DeliveryTable() {
  const [filter, setFilter] = useState<Filter>('all')
  const tally = deliveryTally()

  const rows =
    filter === 'all' ? DELIVERABLES : DELIVERABLES.filter((d) => d.section === filter)

  return (
    <section
      className="rounded-panel p-4"
      style={{
        background: '#17181B',
        border: '1px solid rgba(255,255,255,0.06)',
        boxShadow:
          'inset 0 1px 0 rgba(255,255,255,0.07), 0 1px 2px rgba(0,0,0,0.8), 0 14px 30px rgba(0,0,0,0.5)',
      }}
      aria-label="Expected deliverables"
    >
      <div className="mb-1 flex items-baseline justify-between gap-3">
        <h2 className="text-[14.5px] font-medium text-ink">Delivery table</h2>
        <span className="font-mono text-[11.5px] text-ink-faint">
          {DELIVERABLES.length} requirements
        </span>
      </div>
      <p className="mb-3 font-mono text-[11.5px] leading-relaxed text-ink-faint">
        every expected deliverable in the problem statement, mapped to the section and
        module that satisfies it
      </p>

      {/* status tally */}
      <div className="mb-3 flex flex-wrap gap-2">
        {(['live', 'partial', 'mocked'] as const).map((s) => (
          <span
            key={s}
            className="inline-flex items-center gap-1.5 rounded-[5px] px-2 py-[3px] font-mono text-[11.5px]"
            style={{
              color: STATUS_COLOR[s],
              background: alpha(STATUS_COLOR[s], 0.09),
              border: `1px solid ${alpha(STATUS_COLOR[s], 0.22)}`,
            }}
          >
            <span
              className="h-[4px] w-[4px] rounded-full"
              style={{ background: STATUS_COLOR[s] }}
            />
            {tally[s]} {STATUS_LABEL[s]}
          </span>
        ))}
      </div>

      {/* section filter */}
      <div
        className="mb-3 flex overflow-hidden rounded-[7px]"
        style={{
          background: '#0D0E10',
          border: '1px solid rgba(255,255,255,0.05)',
          boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.9)',
        }}
        role="group"
        aria-label="Filter deliverables by section"
      >
        {FILTERS.map((f) => (
          <button
            key={f}
            type="button"
            onClick={() => setFilter(f)}
            aria-pressed={filter === f}
            className="flex-1 cursor-pointer px-2.5 py-1.5 font-mono text-[11.5px] transition-colors duration-150"
            style={{
              background: filter === f ? 'rgba(255,255,255,0.06)' : 'transparent',
              color: filter === f ? '#E8E9EB' : '#6A6C72',
            }}
          >
            {f === 'all' ? 'all' : f.toLowerCase()}
          </button>
        ))}
      </div>

      {/* the table itself - scrolls horizontally rather than squeezing columns */}
      <div className="console-scroll overflow-x-auto">
        <table className="w-full min-w-[820px] border-collapse text-left">
          <thead>
            <tr>
              {['requirement', 'section', 'how it is delivered', 'module', 'status'].map(
                (h) => (
                  <th
                    key={h}
                    scope="col"
                    className="pb-2 font-mono text-[11px] font-medium tracking-[0.02em] text-ink-faint"
                    style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}
                  >
                    {h}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map((d) => (
              <tr key={d.id}>
                <td
                  className="w-[190px] py-2.5 pr-3 align-top text-[13px] leading-relaxed text-ink"
                  style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                >
                  {d.requirement}
                </td>

                <td
                  className="w-[80px] py-2.5 pr-3 align-top"
                  style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                >
                  <span
                    className="inline-block rounded-[4px] px-1.5 py-[2px] font-mono text-[11px]"
                    style={{
                      color: SECTION_COLOR[d.section],
                      background: alpha(SECTION_COLOR[d.section], 0.09),
                    }}
                  >
                    {d.section}
                  </span>
                </td>

                <td
                  className="py-2.5 pr-3 align-top text-[13px] leading-relaxed text-ink-dim"
                  style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                >
                  {d.delivered}
                </td>

                <td
                  className="w-[150px] py-2.5 pr-3 align-top font-mono text-[11.5px] leading-relaxed text-ink-faint"
                  style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                >
                  {d.module}
                </td>

                <td
                  className="w-[86px] py-2.5 align-top"
                  style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                >
                  <span
                    className="inline-flex items-center gap-1.5 font-mono text-[11.5px]"
                    style={{ color: STATUS_COLOR[d.status] }}
                  >
                    <span
                      className="h-[4px] w-[4px] shrink-0 rounded-full"
                      style={{ background: STATUS_COLOR[d.status] }}
                    />
                    {STATUS_LABEL[d.status]}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p
        className="mt-3.5 pt-3 font-mono text-[11px] leading-relaxed text-ink-faint/80"
        style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
      >
        <span style={{ color: STATUS_COLOR.live }}>implemented</span> means the code runs
        as described.{' '}
        <span style={{ color: STATUS_COLOR.partial }}>partial</span> means the structure
        is real but the numbers are not.{' '}
        <span style={{ color: STATUS_COLOR.mocked }}>mocked</span> means the stage is
        scripted for demonstration - the interface is defined and the swap point is
        documented, but no model is trained.
      </p>
    </section>
  )
}
