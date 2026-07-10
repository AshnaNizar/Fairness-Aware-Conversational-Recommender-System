import { useState, useRef, useEffect } from 'react'
import { Send, RefreshCw, ChevronDown, ChevronRight, BarChart2, X } from 'lucide-react'
import logoSrc   from './assets/logo.jpg'
import avatarSrc from './assets/avatar.jpg'

const API = ''  // same origin when served by FastAPI; use 'http://localhost:7860' for dev

// ── Tag component ─────────────────────────────────────────────────────────────
function Tag({ type }) {
  const styles = {
    relevance: { bg: 'var(--tag-rel-bg)', color: 'var(--tag-rel)' },
    gender:    { bg: 'var(--tag-gen-bg)', color: 'var(--tag-gen)' },
    region:    { bg: 'var(--tag-reg-bg)', color: 'var(--tag-reg)' },
  }
  const labels = { relevance: 'Relevance Pick', gender: 'Gender Diversity', region: 'Regional Diversity' }
  const s = styles[type] || styles.relevance
  return (
    <span style={{
      background: s.bg, color: s.color,
      borderRadius: 'var(--pill)', fontSize: '0.68rem', fontWeight: 600,
      padding: '2px 9px', whiteSpace: 'nowrap', flexShrink: 0,
    }}>
      {labels[type] || 'Relevance Pick'}
    </span>
  )
}

// ── Passive membership tags ───────────────────────────────────────────────────
// Shows that a film belongs to a protected group even when the reranker did NOT
// promote it on that axis (i.e. it earned its slot on relevance). This exposes
// the true fairness composition of a list — e.g. a female-directed film sitting
// at rank 1 on merit — which the amber promotion pills alone would hide.
function isFemale(m)     { return (m.gender || '').toLowerCase() === 'female' }
function isNonWestern(m) { return (m.region || '').toLowerCase().replace('_', '-') === 'non-western' }

function MembershipTags({ movie }) {
  const flag = movie.flag || 'relevance'
  const tags = []
  // female-directed, but NOT already promoted on gender
  if (isFemale(movie) && flag !== 'gender') {
    tags.push({ key: 'gender', label: 'Female-directed',
                color: 'var(--tag-gen)', border: 'var(--tag-gen)' })
  }
  // non-western, but NOT already promoted on region
  if (isNonWestern(movie) && flag !== 'region') {
    tags.push({ key: 'region', label: 'Non-Western',
                color: 'var(--tag-reg)', border: 'var(--tag-reg)' })
  }
  if (tags.length === 0) return null
  return (
    <>
      {tags.map(t => (
        <span key={t.key} style={{
          background: 'transparent', color: t.color,
          border: `1px solid ${t.border}`, opacity: 0.7,
          borderRadius: 'var(--pill)', fontSize: '0.62rem', fontWeight: 600,
          padding: '1px 7px', whiteSpace: 'nowrap', flexShrink: 0,
        }}>
          {t.label}
        </span>
      ))}
    </>
  )
}


// ── Movie list item with collapsible CoT ──────────────────────────────────────
function MovieItem({ movie, rank, userIdx, excludedGenres, includeGenres, fairList, fairFlags }) {
  const [open,    setOpen]    = useState(false)
  const [steps,   setSteps]   = useState(null)   // null = not fetched yet
  const [loading, setLoading] = useState(false)
  const [profile, setProfile] = useState(null)

  async function fetchCot() {
    if (steps !== null) { setOpen(o => !o); return }  // already fetched, just toggle
    setOpen(true)
    setLoading(true)
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), 60000)  // 60s hard cap
    try {
      const r = await fetch(`${API}/cot_explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: ctrl.signal,
        body: JSON.stringify({
          user_idx:        userIdx,
          movie_idx:       movie.id,
          fair_list:       fairList  || [],
          fair_flags:      fairFlags || [],
          excluded_genres: excludedGenres || [],
          include_genres:  includeGenres  || [],
        })
      })
      // Read raw text first so we can debug if JSON parse fails
      const text = await r.text()
      console.log('CoT raw response:', text)
      if (!text || text.trim() === '') {
        setSteps(['Server returned empty response. Check uvicorn terminal for errors.'])
        setLoading(false)
        return
      }
      let data
      try {
        data = JSON.parse(text)
      } catch (parseErr) {
        setSteps([`JSON parse error: ${parseErr.message}`, `Raw response: ${text.slice(0, 200)}`])
        setLoading(false)
        return
      }
      setSteps(data.steps || ['No steps returned.'])
      setProfile(data.profile || null)
    } catch (e) {
      if (e.name === 'AbortError') {
        setSteps(['Reasoning timed out (the LLM backend was too slow). Try again, or the server will fall back to rule-based explanations.'])
      } else {
        setSteps([`Network error: ${e.message}. Is the API server running?`])
      }
    } finally {
      clearTimeout(timer)
      setLoading(false)
    }
  }

  const metaParts = [movie.director, movie.region, movie.year]
    .filter(p => p && p.toLowerCase() !== 'unknown' && p !== 'nan')

  return (
    <div style={{ borderBottom: '1px solid var(--border)', padding: '10px 0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: '0.82rem', fontWeight: 500, marginBottom: 2, lineHeight: 1.3 }}>
            {rank}. {movie.title}
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--muted)', marginBottom: 5 }}>
            {metaParts.slice(0, 3).join(' • ')}
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4, flexShrink: 0 }}>
          <Tag type={movie.flag} />
          <MembershipTags movie={movie} />
        </div>
      </div>

      <button
        onClick={fetchCot}
        style={{
          display: 'flex', alignItems: 'center', gap: 4,
          background: 'none', border: 'none', padding: 0,
          fontSize: '0.68rem', color: 'var(--cot-title)', marginTop: 2,
        }}
      >
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        CoT Trace
      </button>

      {open && (
        <div style={{
          background: 'var(--cot-bg)', borderRadius: 6,
          padding: '10px 12px', marginTop: 6,
        }}>
          <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--cot-title)', marginBottom: 5 }}>
            CoT Trace — <em>{movie.title}</em>
            {profile && (
              <span style={{ color: 'var(--cot-step)', fontWeight: 400, marginLeft: 8 }}>
                · user likes {profile.liked.join(', ') || '—'} · diversity: {profile.diversity}
              </span>
            )}
          </div>
          {loading ? (
            <div style={{ fontSize: '0.68rem', color: 'var(--cot-step)' }}>Loading reasoning...</div>
          ) : (steps || []).map((s, i) => (
            <div key={i} style={{ fontSize: '0.68rem', color: 'var(--cot-step)', lineHeight: 1.7 }}>
              • Step {i + 1}: {s}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Metric table (per-list or cohort) ─────────────────────────────────────────
// "Higher is better" set — everything else improves when it goes DOWN (SPD/EOD
// toward 0, exposure gap toward 0, collapse/gini down). This drives delta color.
const HIGHER_BETTER = new Set([
  'NDCG@10 (avg)', 'Precision@10 (avg)', 'Recall@10 (avg)',
  'Gender rND', 'Region rND', 'Catalog Coverage',
])
const TOWARD_ZERO = new Set([
  'Gender SPD', 'Region SPD', 'Gender EOD', 'Region EOD',
  'Gender Exposure Gap', 'Region Exposure Gap',
])

function fmtNum(v) {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'string') return v
  return v.toFixed(4)
}

function deltaColor(metric, noRerank, fair, delta) {
  if (delta === null || delta === undefined) return 'var(--muted)'
  const GOOD = '#4ADE80', BAD = '#F87171', NEUTRAL = 'var(--muted)'
  if (Math.abs(delta) < 1e-9) return NEUTRAL
  if (HIGHER_BETTER.has(metric)) return delta > 0 ? GOOD : BAD
  if (TOWARD_ZERO.has(metric)) {
    // improved iff |fair| < |no_rerank|
    if (fair === null || noRerank === null) return NEUTRAL
    return Math.abs(fair) < Math.abs(noRerank) ? GOOD : BAD
  }
  // collapse / gini: lower is better
  return delta < 0 ? GOOD : BAD
}

function MetricTable({ title, subtitle, rows, muted }) {
  if (!rows || rows.length === 0) return null
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
        <span style={{ fontSize: '0.78rem', fontWeight: 700, color: muted ? 'var(--muted)' : 'var(--text)' }}>
          {title}
        </span>
      </div>
      {subtitle && (
        <div style={{ fontSize: '0.64rem', color: 'var(--muted)', marginBottom: 8 }}>{subtitle}</div>
      )}
      {/* header */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1.35fr 0.9fr 0.9fr 0.8fr',
        gap: 4, fontSize: '0.6rem', color: 'var(--muted)',
        textTransform: 'uppercase', letterSpacing: '0.03em',
        paddingBottom: 4, borderBottom: '1px solid var(--border)',
      }}>
        <span>Metric</span>
        <span style={{ textAlign: 'right' }}>No-rerank</span>
        <span style={{ textAlign: 'right' }}>FA*IR</span>
        <span style={{ textAlign: 'right' }}>Δ</span>
      </div>
      {rows.map(r => (
        <div key={r.metric} style={{
          display: 'grid', gridTemplateColumns: '1.35fr 0.9fr 0.9fr 0.8fr',
          gap: 4, fontSize: '0.68rem', padding: '5px 0',
          borderBottom: '1px solid rgba(255,255,255,0.04)', alignItems: 'center',
        }}>
          <span style={{ color: 'var(--muted)' }}>{r.metric}</span>
          <span style={{ textAlign: 'right', color: 'var(--text)', fontVariantNumeric: 'tabular-nums' }}>
            {fmtNum(r.no_rerank)}
          </span>
          <span style={{ textAlign: 'right', color: 'var(--text)', fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
            {fmtNum(r.fair)}
          </span>
          <span style={{
            textAlign: 'right', fontWeight: 700, fontVariantNumeric: 'tabular-nums',
            color: deltaColor(r.metric, r.no_rerank, r.fair, r.delta),
          }}>
            {r.delta === null || r.delta === undefined ? '—'
              : (r.delta > 0 ? '+' : '') + r.delta.toFixed(4)}
          </span>
        </div>
      ))}
    </div>
  )
}

// ── Analytics modal ───────────────────────────────────────────────────────────
function AnalyticsModal({ chartB64, metrics, onClose }) {
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      background: 'rgba(0,0,0,0.75)', display: 'flex',
      alignItems: 'center', justifyContent: 'center',
    }} onClick={onClose}>
      <div style={{
        background: '#1C1C1C', borderRadius: 14, padding: 24,
        maxWidth: 900, width: '90%', maxHeight: '88vh',
        overflow: 'auto', position: 'relative',
        border: '1px solid var(--border)',
      }} onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <span style={{ fontWeight: 700, fontSize: '1rem' }}>Complete Analytics</span>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--muted)', padding: 4 }}>
            <X size={18} />
          </button>
        </div>

        {/* Chart first, then the full metric tables */}
        {chartB64 && (
          <img
            src={`data:image/png;base64,${chartB64}`}
            alt="Analytics"
            style={{ width: '100%', borderRadius: 8, marginBottom: 20 }}
          />
        )}

        {metrics?.table ? (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
            {/* Per-list metrics */}
            <MetricTable
              title="Per-list metrics"
              subtitle="Computed from this user's top-10"
              rows={metrics.table.filter(r => r.scope !== 'cohort')}
            />
            {/* Cohort metrics */}
            <div>
              <MetricTable
                title="Cohort metrics"
                subtitle={`Over ${metrics.cohort?.fair?.n_users ?? 0} users`
                  + ` (${metrics.cohort?.fair?.n_seed ?? 0} warm-up + ${metrics.cohort?.fair?.n_live ?? 0} live)`
                  + ` · accuracy over ${metrics.cohort?.fair?.n_scored_users ?? 0} with ground truth`}
                rows={metrics.table.filter(r => r.scope === 'cohort')}
                
              />
              <div style={{ fontSize: '0.6rem', color: 'var(--muted)', marginTop: 4, lineHeight: 1.5, opacity: 0.8 }}>
                Accuracy (NDCG/Precision/Recall) is a session average — per-user
                values are near-binary, so they converge here toward the model's
                true ~0.011 as more users load.
              </div>
            </div>
          </div>
        ) : (
          /* fallback for older metric shape */
          <div style={{ display: 'flex', gap: 12 }}>
            {[
              { label: 'SPD',    val: metrics?.spd,  color: '#4ADE80' },
              { label: 'OEAD',   val: metrics?.oead, color: '#F3A425' },
              { label: 'NDCG@10',val: metrics?.ndcg, color: '#82AAFF' },
            ].map(m => (
              <div key={m.label} style={{
                background: 'var(--bg)', borderRadius: 8, padding: '8px 16px',
                border: '1px solid var(--border)',
              }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--muted)', marginBottom: 2 }}>{m.label}</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: m.color }}>{m.val}</div>
              </div>
            ))}
          </div>
        )}

        {/* Operating point */}
        {metrics?.targets && (
          <div style={{
            marginTop: 20, padding: '10px 12px', borderRadius: 8,
            background: 'rgba(255,255,255,0.03)', fontSize: '0.7rem',
            color: 'var(--muted)', lineHeight: 1.5,
          }}>
            FA*IR p={metrics.targets.p_gender}, α={metrics.targets.alpha}
            {' · '}catalogue supply {metrics.targets.pct_female_catalog}% female-directed,{' '}
            {metrics.targets.pct_nonwestern_catalog}% non-western
          </div>
        )}
      </div>
    </div>
  )
}

// ── Chat message bubble ───────────────────────────────────────────────────────
// Parse bot message content into structured segments:
// numbered lists, bullet lists, bold text, and plain paragraphs
function parseContent(content) {
  if (!content) return []
  const segments = []
  const lines = content.split('\n')
  let i = 0

  while (i < lines.length) {
    const line = lines[i].trim()
    if (!line) { i++; continue }

    // Numbered list: "1. Item" or "1) Item"
    if (/^\d+[.)\s]/.test(line)) {
      const items = []
      while (i < lines.length && /^\d+[.)\s]/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+[.)\s]\s*/, ''))
        i++
      }
      segments.push({ type: 'numbered', items })
      continue
    }

    // Bullet list: "- Item" or "• Item" or "* Item"
    if (/^[-•*]\s/.test(line)) {
      const items = []
      while (i < lines.length && /^[-•*]\s/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-•*]\s+/, ''))
        i++
      }
      segments.push({ type: 'bullets', items })
      continue
    }

    // Plain paragraph
    segments.push({ type: 'text', text: line })
    i++
  }
  return segments
}

// Render inline bold: **text**
function renderInline(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/)
  return parts.map((p, i) =>
    p.startsWith('**') && p.endsWith('**')
      ? <strong key={i} style={{ color: 'var(--text)', fontWeight: 600 }}>{p.slice(2, -2)}</strong>
      : <span key={i}>{p}</span>
  )
}

function ChatBubble({ role, content }) {
  const isUser = role === 'user'
  const segments = isUser ? null : parseContent(content)

  return (
    <div style={{
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      marginBottom: 12, padding: '0 4px',
    }}>
      <div style={{
        maxWidth: '78%',
        background: isUser ? '#2A2A2A' : 'transparent',
        borderRadius: isUser ? '18px 18px 4px 18px' : '0',
        padding: isUser ? '10px 14px' : '4px 0',
        fontSize: '0.88rem',
        lineHeight: 1.55,
        color: 'var(--text)',
      }}>
        {isUser ? content : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {(segments || []).map((seg, si) => {
              if (seg.type === 'numbered') return (
                <ol key={si} style={{ paddingLeft: 20, margin: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {seg.items.map((item, ii) => (
                    <li key={ii} style={{
                      fontSize: '0.86rem', lineHeight: 1.5, color: 'var(--text)',
                      paddingLeft: 4,
                    }}>
                      {renderInline(item)}
                    </li>
                  ))}
                </ol>
              )
              if (seg.type === 'bullets') return (
                <ul key={si} style={{ paddingLeft: 18, margin: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {seg.items.map((item, ii) => (
                    <li key={ii} style={{
                      fontSize: '0.86rem', lineHeight: 1.5, color: 'var(--text)',
                      paddingLeft: 4, listStyleType: 'disc',
                    }}>
                      {renderInline(item)}
                    </li>
                  ))}
                </ul>
              )
              return (
                <p key={si} style={{ margin: 0, fontSize: '0.88rem', lineHeight: 1.55 }}>
                  {renderInline(seg.text)}
                </p>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Inline recommendation card (shown in chat for recommend intent) ───────────
function RecommendCard({ movie, cotData, rank }) {
  const [open, setOpen] = useState(false)
  const metaParts = [movie.director, movie.region, movie.year]
    .filter(p => p && p.toLowerCase() !== 'unknown' && p !== 'nan')

  return (
    <div style={{
      background: 'var(--cot-bg)', borderRadius: 8,
      padding: '10px 14px', marginBottom: 6,
      border: '1px solid rgba(130,170,255,0.12)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: '0.84rem', fontWeight: 600, color: 'var(--cot-title)', marginBottom: 2 }}>
            {rank}. {movie.title}
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--muted)' }}>
            {metaParts.join(' • ')}
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4, flexShrink: 0 }}>
          <Tag type={movie.flag} />
          <MembershipTags movie={movie} />
        </div>
      </div>

      {cotData && (
        <>
          <button
            onClick={() => setOpen(o => !o)}
            style={{
              display: 'flex', alignItems: 'center', gap: 4, marginTop: 6,
              background: 'none', border: 'none', padding: 0,
              fontSize: '0.68rem', color: 'var(--cot-title)', cursor: 'pointer',
            }}
          >
            {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
            CoT Trace
          </button>
          {open && (
            <div style={{ marginTop: 6 }}>
              {cotData.steps.map((s, i) => (
                <div key={i} style={{ fontSize: '0.68rem', color: 'var(--cot-step)', lineHeight: 1.7 }}>
                  • Step {i + 1}: {s}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}

// ── Comparison modal (baseline vs fair) ───────────────────────────────────────
function CompareModal({ baseMovies, fairMovies, onClose }) {
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      background: 'rgba(0,0,0,0.75)', display: 'flex',
      alignItems: 'center', justifyContent: 'center',
    }} onClick={onClose}>
      <div style={{
        background: '#1C1C1C', borderRadius: 14, padding: 24,
        maxWidth: 860, width: '92%', maxHeight: '88vh',
        overflow: 'auto', border: '1px solid var(--border)',
        position: 'relative',
      }} onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <span style={{ fontWeight: 700, fontSize: '1rem' }}>Baseline vs FA★IR Comparison</span>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--muted)', padding: 4 }}>
            <X size={18} />
          </button>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
          {[['Baseline (Relevance only)', baseMovies, 'relevance'],
            ['FA★IR Reranked (p=0.3)', fairMovies, null]
          ].map(([title, movies, forceFlag]) => (
            <div key={title}>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 10 }}>{title}</div>
              {(movies || []).map((m, i) => (
                <div key={m.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '7px 0', borderBottom: '1px solid var(--border)', gap: 8 }}>
                  <div>
                    <div style={{ fontSize: '0.78rem', fontWeight: 500 }}>{i + 1}. {m.title}</div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--muted)' }}>{m.director} • {m.region}</div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4, flexShrink: 0 }}>
                    <Tag type={forceFlag || m.flag} />
                    <MembershipTags movie={forceFlag ? { ...m, flag: forceFlag } : m} />
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [userStats,    setUserStats]    = useState(null)
  const [userIdx,      setUserIdx]      = useState(null)
  const [fairMovies,   setFairMovies]   = useState([])
  const [baseMovies,   setBaseMovies]   = useState([])
  const [fairList,     setFairList]     = useState([])
  const [fairFlags,    setFairFlags]    = useState([])
  const [exclGenres,   setExclGenres]   = useState([])
  const [inclGenres,   setInclGenres]   = useState([])
  const [metrics,      setMetrics]      = useState(null)
  const [chartB64,     setChartB64]     = useState(null)
  const [messages,     setMessages]     = useState([])
  const [input,        setInput]        = useState('')
  const [loading,      setLoading]      = useState(false)
  const [userLoading,  setUserLoading]  = useState(false)
  const [showAnalytics,setShowAnalytics]= useState(false)
  const [showCompare,  setShowCompare]  = useState(false)
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function loadRandomUser() {
    setUserLoading(true)
    setMessages([])
    try {
      const r   = await fetch(`${API}/random_user`)
      const data = await r.json()
      setUserStats(data.stats)
      setUserIdx(data.user_idx)
      setFairMovies(data.fair_movies)
      setBaseMovies(data.base_movies)
      setFairList(data.fair_movies.map(m => m.id))
      setFairFlags(data.fair_movies.map(m => m.flag))
      setMetrics(data.metrics)
      setChartB64(data.chart_b64)
      setExclGenres([])
      setInclGenres([])
      setMessages([{
        role: 'assistant',
        content: `I've loaded User #${data.stats.id}. They enjoy ${data.stats.top_genres} and have rated ${data.stats.n_rated} films. Ask me to recommend something or explain why a pick appeared!`
      }])
    } catch (e) {
      setMessages([{ role: 'assistant', content: `Error: ${e.message}` }])
    }
    setUserLoading(false)
  }

  async function sendMessage() {
    if (!input.trim() || loading || userIdx === null) return
    const userMsg = input.trim()
    setInput('')
    setMessages(m => [...m, { role: 'user', content: userMsg }])
    setLoading(true)

    // Show typing indicator
    setMessages(m => [...m, { role: 'assistant', content: '…', typing: true }])

    try {
      const r = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMsg,
          user_idx: userIdx,
          history: messages.filter(m => !m.typing).slice(-6).map(m => ({ role: m.role, content: m.content })),
          excluded_genres: exclGenres,
          include_genres:  inclGenres,
          fair_list: fairList,
          fair_flags: fairFlags,
        })
      })
      const data = await r.json()

      setMessages(m => {
        const filtered = m.filter(x => !x.typing)
        return [...filtered, {
          role: 'assistant',
          content: data.reply,
          intent: data.intent,
          cotList: data.cot_list || [],
          fairMovies: data.fair_movies || [],
        }]
      })
      setFairMovies(data.fair_movies)
      setBaseMovies(data.base_movies)
      setFairList(data.fair_list)
      setFairFlags(data.fair_flags)
      setExclGenres(data.excluded_genres)
      setInclGenres(data.include_genres)
      setMetrics(data.metrics)
      setChartB64(data.chart_b64)
    } catch (e) {
      setMessages(m => {
        const filtered = m.filter(x => !x.typing)
        return [...filtered, { role: 'assistant', content: `Error: ${e.message}` }]
      })
    }
    setLoading(false)
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
  }

  const hasUser = userStats !== null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>

      {/* ── Nav ── */}
      <nav style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 24px', height: 52, borderBottom: '1px solid var(--border)',
        background: 'var(--side)', flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <img src={logoSrc} alt="logo" style={{ width: 28, height: 28, borderRadius: 7 }} />
          <span style={{ fontSize: '0.88rem', fontWeight: 600 }}>Movie RecSys</span>
        </div>
        <span style={{ fontSize: '0.88rem', fontWeight: 600 }}>
          Conversational Fairness Aware Recommendation System — Movies
        </span>
        <span style={{ fontSize: '0.82rem', color: 'var(--muted)' }}>Results</span>
      </nav>

      {/* ── 3 columns ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '248px 1fr 240px', flex: 1, overflow: 'hidden' }}>

        {/* ── LEFT: User sidebar ── */}
        <aside style={{
          background: 'var(--side)', borderRight: '1px solid var(--border)',
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}>
          {/* User persona header */}
          <div style={{ padding: '14px 16px 10px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <img src={avatarSrc} alt="persona" style={{ width: 24, height: 24, borderRadius: '50%' }} />
              <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
                User Persona
              </span>
            </div>

            {hasUser ? (
              <>
                <div style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 8 }}>
                  User &nbsp;<span style={{ fontWeight: 700 }}>#{userStats.id}</span>
                </div>
                {[
                  ['Top Genres',       userStats.top_genres],
                  ['Movies Rated',     userStats.n_rated],
                  ['Avg Movie Rating', userStats.avg_rating],
                ].map(([label, val]) => (
                  <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', fontSize: '0.75rem' }}>
                    <span style={{ color: 'var(--muted)' }}>{label}:</span>
                    <span style={{ fontWeight: 500, textAlign: 'right', maxWidth: '55%', wordBreak: 'break-word' }}>{val}</span>
                  </div>
                ))}
              </>
            ) : (
              <div style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>No user loaded yet.</div>
            )}
          </div>

          {/* New random user button */}
          <div style={{ padding: '10px 16px', flexShrink: 0 }}>
            <button
              onClick={loadRandomUser}
              disabled={userLoading}
              style={{
                width: '100%', background: 'var(--accent)', border: 'none',
                borderRadius: 'var(--pill)', color: '#fff', fontSize: '0.78rem',
                fontWeight: 600, padding: '8px 0', display: 'flex',
                alignItems: 'center', justifyContent: 'center', gap: 6,
                opacity: userLoading ? 0.7 : 1,
              }}
            >
              <RefreshCw size={13} />
              {userLoading ? 'Loading...' : 'New Random User'}
            </button>
          </div>

          {/* Movie picks list */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '0 16px 16px' }}>
            <div style={{
              fontSize: '0.68rem', fontWeight: 600, color: 'var(--muted)',
              textTransform: 'uppercase', letterSpacing: '0.08em',
              padding: '8px 0 6px', borderBottom: '1px solid var(--border)', marginBottom: 4,
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            }}>
              <span>{hasUser ? `Default Picks for User #${userStats.id}` : 'Default Picks'}</span>
              {hasUser && (
                <button
                  onClick={() => setShowCompare(true)}
                  style={{ background: 'none', border: 'none', color: 'var(--cot-title)', fontSize: '0.65rem', fontWeight: 600 }}
                >
                  View
                </button>
              )}
            </div>

            {fairMovies.length === 0 ? (
              <p style={{ fontSize: '0.72rem', color: 'var(--muted)', paddingTop: 10 }}>
                No recommendations yet. Load a user to begin.
              </p>
            ) : (
              fairMovies.map((m, i) => <MovieItem key={m.id} movie={m} rank={i + 1} userIdx={userIdx} excludedGenres={exclGenres} includeGenres={inclGenres} fairList={fairMovies.map(x => x.id)} fairFlags={fairFlags} />)
            )}
          </div>
        </aside>

        {/* ── CENTRE: Chat ── */}
        <main style={{
          display: 'flex', flexDirection: 'column',
          background: 'var(--bg)', borderRight: '1px solid var(--border)',
          overflow: 'hidden',
        }}>
          {/* Messages */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '20px 28px' }}>
            {messages.length === 0 ? (
              /* Landing screen */
              <div style={{
                height: '100%', display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center', gap: 16,
              }}>
                <img src={logoSrc} alt="logo" style={{ width: 76, height: 76, borderRadius: 18 }} />
                <div style={{
                  fontSize: '1.75rem', fontWeight: 700,
                  background: 'linear-gradient(135deg, #C084FC, #60A5FA)',
                  WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text', textAlign: 'center',
                }}>
                  What would you like to watch today?
                </div>
                {!hasUser && (
                  <div style={{ fontSize: '0.82rem', color: 'var(--muted)' }}>
                    Load a user with the button on the left to begin.
                  </div>
                )}
              </div>
            ) : (
              <>
                {messages.map((msg, i) => (
                  <div key={i}>
                    <ChatBubble role={msg.role} content={msg.content} />
                    {msg.intent === 'recommend' && msg.fairMovies && msg.fairMovies.length > 0 && (
                      <div style={{ padding: '0 4px', marginTop: -4, marginBottom: 12 }}>
                        {msg.fairMovies.map((movie, mi) => {
                          const cotData = (msg.cotList || []).find(c => c.movie_idx === movie.id)
                          return <RecommendCard key={movie.id} movie={movie} cotData={cotData} rank={mi + 1} />
                        })}
                      </div>
                    )}
                  </div>
                ))}
                <div ref={chatEndRef} />
              </>
            )}
          </div>

          {/* Input bar */}
          <div style={{
            padding: '12px 20px', borderTop: '1px solid var(--border)',
            display: 'flex', gap: 10, alignItems: 'flex-end',
            background: 'var(--bg)', flexShrink: 0,
          }}>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder={hasUser ? 'Suggest ten action movies...' : 'Load a user first...'}
              disabled={!hasUser || loading}
              rows={1}
              style={{
                flex: 1, background: '#232323', border: '1px solid var(--border)',
                borderRadius: 22, color: 'var(--text)', fontSize: '0.88rem',
                padding: '11px 18px', resize: 'none', outline: 'none',
                lineHeight: 1.4,
              }}
            />
            <button
              onClick={sendMessage}
              disabled={!hasUser || loading || !input.trim()}
              style={{
                background: 'var(--accent)', border: 'none', borderRadius: '50%',
                width: 40, height: 40, display: 'flex', alignItems: 'center',
                justifyContent: 'center', flexShrink: 0, color: '#fff',
                opacity: (!hasUser || loading || !input.trim()) ? 0.4 : 1,
                transition: 'opacity 0.2s',
              }}
            >
              <Send size={16} />
            </button>
          </div>
        </main>

        {/* ── RIGHT: Metrics & analytics ── */}
        <aside style={{
          background: 'var(--side)', display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
        }}>
          <div style={{ padding: '14px 16px 10px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 600 }}>Results</span>
              {metrics && (
                <BarChart2 size={15} color="var(--muted)" style={{ cursor: 'pointer' }}
                  onClick={() => setShowAnalytics(true)} />
              )}
            </div>
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '10px 16px' }}>
            {!metrics ? (
              <div style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center',
                justifyContent: 'center', height: '100%', gap: 10, textAlign: 'center',
              }}>
                <div style={{ fontSize: '1.4rem', color: 'var(--muted)' }}>ⓘ</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--muted)', lineHeight: 1.5 }}>
                  Start a new conversation to view recommendation analytics
                </div>
              </div>
            ) : (
              <>
                {/* ── Per-list metrics (this recommendation) ── */}
                <MetricTable
                  title="Per-list metrics"
                  subtitle="Computed from this user's top-10"
                  rows={(metrics.table || []).filter(r => r.scope !== 'cohort')}
                />

                {/* ── Cohort metrics (across users this session) ── */}
                <MetricTable
                  title="Cohort metrics"
                  subtitle={`Over ${metrics.cohort?.fair?.n_users ?? 0} users`
                    + ` (${metrics.cohort?.fair?.n_seed ?? 0} warm-up + ${metrics.cohort?.fair?.n_live ?? 0} live)`
                    + ` · accuracy over ${metrics.cohort?.fair?.n_scored_users ?? 0} with ground truth`}
                  rows={(metrics.table || []).filter(r => r.scope === 'cohort')}
                  
                />
                <div style={{ fontSize: '0.6rem', color: 'var(--muted)', marginTop: -8, marginBottom: 12, lineHeight: 1.5, opacity: 0.8 }}>
                  Accuracy (NDCG/Precision/Recall) is a session average — per-user
                  values are near-binary and misleading, so they converge here toward
                  the model's true ~0.011 as more users load.
                </div>

                {/* Operating point */}
                {metrics.targets && (
                  <div style={{
                    marginTop: 10, padding: '8px 10px', borderRadius: 8,
                    background: 'rgba(255,255,255,0.03)', fontSize: '0.68rem',
                    color: 'var(--muted)', lineHeight: 1.5,
                  }}>
                    FA*IR p={metrics.targets.p_gender}, α={metrics.targets.alpha}
                    {' · '}supply {metrics.targets.pct_female_catalog}% F-dir,{' '}
                    {metrics.targets.pct_nonwestern_catalog}% non-western
                  </div>
                )}

                {/* Mini chart preview */}
                {chartB64 && (
                  <div style={{ marginTop: 14 }}>
                    <img
                      src={`data:image/png;base64,${chartB64}`}
                      alt="Chart preview"
                      style={{ width: '100%', borderRadius: 8, cursor: 'pointer' }}
                      onClick={() => setShowAnalytics(true)}
                    />
                  </div>
                )}

                <button
                  onClick={() => setShowAnalytics(true)}
                  style={{
                    width: '100%', marginTop: 12, background: 'var(--accent)',
                    border: 'none', borderRadius: 'var(--pill)', color: '#fff',
                    fontSize: '0.75rem', fontWeight: 600, padding: '8px 0',
                  }}
                >
                  View Complete Analytics
                </button>
              </>
            )}
          </div>
        </aside>
      </div>

      {/* ── Modals ── */}
      {showAnalytics && (
        <AnalyticsModal
          chartB64={chartB64}
          metrics={metrics}
          onClose={() => setShowAnalytics(false)}
        />
      )}
      {showCompare && (
        <CompareModal
          baseMovies={baseMovies}
          fairMovies={fairMovies}
          onClose={() => setShowCompare(false)}
        />
      )}
    </div>
  )
}