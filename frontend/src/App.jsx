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

// ── Movie list item with collapsible CoT ──────────────────────────────────────
function MovieItem({ movie, rank, userIdx, excludedGenres, includeGenres }) {
  const [open,    setOpen]    = useState(false)
  const [steps,   setSteps]   = useState(null)   // null = not fetched yet
  const [loading, setLoading] = useState(false)
  const [profile, setProfile] = useState(null)

  async function fetchCot() {
    if (steps !== null) { setOpen(o => !o); return }  // already fetched, just toggle
    setOpen(true)
    setLoading(true)
    try {
      const r = await fetch(`${API}/cot_explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_idx:        userIdx,
          movie_idx:       movie.id,
          fair_list:       [],
          fair_flags:      [],
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
      setSteps([`Network error: ${e.message}`])
    }
    setLoading(false)
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
        <Tag type={movie.flag} />
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

        {/* Metric pills */}
        <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
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

        {chartB64 && (
          <img
            src={`data:image/png;base64,${chartB64}`}
            alt="Analytics"
            style={{ width: '100%', borderRadius: 8 }}
          />
        )}
      </div>
    </div>
  )
}

// ── Chat message bubble ───────────────────────────────────────────────────────
function ChatBubble({ role, content }) {
  const isUser = role === 'user'
  return (
    <div style={{
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      marginBottom: 12, padding: '0 4px',
    }}>
      <div style={{
        maxWidth: '72%',
        background: isUser ? '#2A2A2A' : 'transparent',
        borderRadius: isUser ? '18px 18px 4px 18px' : '0',
        padding: isUser ? '10px 14px' : '4px 0',
        fontSize: '0.88rem',
        lineHeight: 1.55,
        color: 'var(--text)',
      }}>
        {content}
      </div>
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
                  <Tag type={forceFlag || m.flag} />
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
        return [...filtered, { role: 'assistant', content: data.reply }]
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
              <img src={avatarSrc} alt="persona" style={{ width: 34, height: 34, borderRadius: '50%' }} />
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
              fairMovies.map((m, i) => <MovieItem key={m.id} movie={m} rank={i + 1} userIdx={userIdx} excludedGenres={exclGenres} includeGenres={inclGenres} />)
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
                  <ChatBubble key={i} role={msg.role} content={msg.content} />
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
                {/* Metric rows */}
                {[
                  { label: 'SPD',     val: metrics.spd,  color: '#4ADE80' },
                  { label: 'OEAD',    val: metrics.oead, color: '#F3A425' },
                  { label: 'NDCG@10', val: metrics.ndcg, color: '#82AAFF' },
                ].map(m => (
                  <div key={m.label} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    padding: '9px 0', borderBottom: '1px solid var(--border)', fontSize: '0.78rem',
                  }}>
                    <span style={{ color: 'var(--muted)' }}>{m.label}</span>
                    <span style={{ color: m.color, fontWeight: 700 }}>{m.val}</span>
                  </div>
                ))}

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
