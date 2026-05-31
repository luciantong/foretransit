import { useEffect, useMemo, useRef, useState } from 'react'
import { MapContainer, Marker, Popup, TileLayer, useMap, useMapEvents } from 'react-leaflet'
import { divIcon } from 'leaflet'
import './App.css'

const API_ROOT = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/+$/, '')
const MAP_RENDER_START_ZOOM = 14
const MAP_RENDER_FADE_START_ZOOM = 12.0
const MAP_RENDER_FADE_END_ZOOM = 14.4
const TRANSIT_MODE_OPTIONS = [
  { value: 'railway', label: 'Train stations' },
  { value: 'bus', label: 'Bus stops' },
  { value: 'bike', label: 'Bikeshare stations' },
]

function buildApiUrl(path) {
  if (/^https?:\/\//i.test(path)) {
    return path
  }
  return `${API_ROOT}${path.startsWith('/') ? path : `/${path}`}`
}

async function fetchJson(path) {
  const endpoint = buildApiUrl(path)
  const res = await fetch(endpoint, {
    headers: {
      Accept: 'application/json',
    },
  })

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`)
  }

  const contentType = res.headers.get('content-type') || ''
  if (!contentType.toLowerCase().includes('application/json')) {
    const preview = (await res.text()).slice(0, 120).replace(/\s+/g, ' ').trim()
    if (preview.startsWith('<')) {
      throw new Error(
        `Backend returned HTML instead of JSON for ${path}. Start API on 127.0.0.1:8000 and use frontend dev server proxy, or set VITE_API_BASE_URL=http://127.0.0.1:8000.`,
      )
    }
    throw new Error(`Expected JSON response for ${path}, got content-type: ${contentType || 'unknown'}`)
  }

  return res.json()
}

function describeFetchError(error, endpointLabel) {
  if (error instanceof TypeError) {
    return `Could not reach backend (${endpointLabel}). Start API server on port 8000 and retry.`
  }
  if (error instanceof Error) {
    return error.message
  }
  return `Request failed for ${endpointLabel}`
}

function formatModelName(name) {
  if (!name) {
    return 'none'
  }
  if (name === 'xgboost') {
    return 'XGBoost'
  }
  return name.charAt(0).toUpperCase() + name.slice(1)
}

function modeStyle(mode) {
  if (mode === 'railway') {
    return 'railway'
  }
  if (mode === 'light_rail' || mode === 'light-rail') {
    return 'railway'
  }
  if (mode === 'bike') {
    return 'bike'
  }
  return 'bus'
}

function normalizeTransitFilterMode(mode) {
  if (mode === 'light_rail' || mode === 'light-rail') {
    return 'railway'
  }
  return mode
}

function edgeFadeOpacity(stop, bounds) {
  if (!bounds) {
    return 1
  }

  const north = bounds.getNorth()
  const south = bounds.getSouth()
  const east = bounds.getEast()
  const west = bounds.getWest()

  const latSpan = Math.max(0.00001, north - south)
  const lonSpan = Math.max(0.00001, east - west)

  const edgeZoneLat = latSpan * 0.12
  const edgeZoneLon = lonSpan * 0.12

  const distToLatEdge = Math.min(stop.lat - south, north - stop.lat)
  const distToLonEdge = Math.min(stop.lon - west, east - stop.lon)

  const latOpacity = Math.min(1, Math.max(0, distToLatEdge / edgeZoneLat))
  const lonOpacity = Math.min(1, Math.max(0, distToLonEdge / edgeZoneLon))

  return 0.22 + 0.78 * Math.min(latOpacity, lonOpacity)
}

function zoomRevealOpacity(zoom) {
  if (typeof zoom !== 'number' || Number.isNaN(zoom)) {
    return 0
  }
  if (zoom <= MAP_RENDER_FADE_START_ZOOM) {
    return 0
  }
  if (zoom >= MAP_RENDER_FADE_END_ZOOM) {
    return 1
  }

  const t = (zoom - MAP_RENDER_FADE_START_ZOOM) / (MAP_RENDER_FADE_END_ZOOM - MAP_RENDER_FADE_START_ZOOM)
  // Ease-out curve so stops become readable quickly once zooming in.
  return 1 - (1 - t) * (1 - t)
}

function colorByScore(score) {
  if (typeof score !== 'number' || !Number.isFinite(score)) {
    return '#000000'
  }
  if (score < 50) {
    return '#cc0000'
  }
  if (score < 90) {
    return '#cca300'
  }
  return '#1a573c'
}

function colorByBikeAvailability(bikesAvailable) {
  if (typeof bikesAvailable !== 'number' || Number.isNaN(bikesAvailable)) {
    return '#64748b'
  }
  if (bikesAvailable <= 2) {
    return '#dc2626'
  }
  if (bikesAvailable <= 6) {
    return '#eab308'
  }
  return '#16a34a'
}

function factorToneClass(factor) {
  if (factor?.tone === 'positive' || (typeof factor?.score_100 === 'number' && factor.score_100 >= 70)) {
    return 'factor-item--positive'
  }
  if (factor?.tone === 'negative' || (typeof factor?.score_100 === 'number' && factor.score_100 <= 44)) {
    return 'factor-item--negative'
  }
  return 'factor-item--neutral'
}

function factorValueLabel(factor) {
  if (typeof factor?.display_value === 'string' && factor.display_value.trim()) {
    return factor.display_value
  }
  if (typeof factor?.factor === 'string' && factor.factor.trim()) {
    return factor.factor
  }
  return 'loading'
}

function confidenceLabel(score) {
  if (typeof score !== 'number' || !Number.isFinite(score)) {
    return 'Confidence: loading'
  }
  if (score >= 70) {
    return 'Confidence: High'
  }
  if (score >= 45) {
    return 'Confidence: Medium'
  }
  return 'Confidence: Low'
}

function compactConfidence(score) {
  if (typeof score !== 'number' || !Number.isFinite(score)) {
    return 'loading'
  }
  if (score >= 70) {
    return 'HIGH'
  }
  if (score >= 45) {
    return 'MED'
  }
  return 'LOW'
}

function factorRowTitle(factor) {
  const raw = String(factor?.factor || '').toLowerCase()
  if (raw.includes('boarding') || raw.includes('dwell') || raw.includes('delay')) {
    return 'BOARDING DELAYS'
  }
  if (raw.includes('traffic') || raw.includes('leg')) {
    return 'TRAFFIC CONGESTION'
  }
  if (raw.includes('route') || raw.includes('stops') || raw.includes('gap') || raw.includes('section')) {
    return 'ROUTE SEGMENT'
  }
  return raw ? raw.toUpperCase() : 'LIVE FACTOR'
}

function factorBadgeLabel(factor) {
  const value = factorValueLabel(factor).toLowerCase()
  if (value === 'no delay' || value === 'on schedule') {
    return 'no delays'
  }
  if (value === 'slight delay' || value === 'short wait') {
    return 'minor delays'
  }
  if (value === 'moderate delay' || value === 'long wait') {
    return 'moderate'
  }
  if (value === 'heavy delay') {
    return 'major delays'
  }
  if (value === 'few stops' || value === 'several stops' || value === 'many stops') {
    return value
  }
  return value || 'loading'
}

function serviceStatusFromScore(score) {
  if (typeof score !== 'number' || !Number.isFinite(score)) {
    return 'Unknown'
  }
  if (score >= 90) {
    return 'Optimized'
  }
  if (score >= 70) {
    return 'Stable'
  }
  if (score >= 50) {
    return 'Watch'
  }
  return 'Disrupted'
}

function adviceFromScore(score) {
  if (typeof score !== 'number' || !Number.isFinite(score)) {
    return 'loading.'
  }
  if (score >= 90) {
    return 'Service is running on time. No need to rush.'
  }
  if (score >= 50) {
    return 'Minor traffic detected. You might have a 5-minute wait.'
  }
  return 'Major delays. Check for a nearby subway or ride-share.'
}

function transitTagLabel(mode) {
  if (modeStyle(mode) === 'railway') {
    return 'Train'
  }
  if (modeStyle(mode) === 'bike') {
    return 'Bike'
  }
  return 'Bus'
}

function scoreBand(score) {
  if (score >= 70) {
    return { score_100: score, tone: 'positive', impact: 'low' }
  }
  if (score <= 44) {
    return { score_100: score, tone: 'negative', impact: 'high' }
  }
  return { score_100: score, tone: 'neutral', impact: 'medium' }
}

function buildModelConsideration(inputKey, inputValue) {
  switch (inputKey) {
    case 'delay_seconds': {
      const delay = Number(inputValue || 0)
      if (delay < 60) return { factor: 'No current boarding delays', display_value: 'No delay', ...scoreBand(86) }
      if (delay < 180) return { factor: 'Minor boarding slowdown detected', display_value: 'Slight delay', ...scoreBand(62) }
      if (delay < 300) return { factor: 'Boarding delays are building', display_value: 'Moderate delay', ...scoreBand(42) }
      return { factor: 'Major boarding delays right now', display_value: 'Heavy delay', ...scoreBand(24) }
    }
    case 'gap_seconds': {
      const gap = Number(inputValue || 0)
      if (gap <= 120) return { factor: 'Vehicles are arriving consistently', display_value: 'On schedule', ...scoreBand(82) }
      if (gap <= 420) return { factor: 'Small spacing gaps between vehicles', display_value: 'Short wait', ...scoreBand(58) }
      return { factor: 'Long wait gap between vehicles', display_value: 'Long wait', ...scoreBand(34) }
    }
    case 'cumulative_dwell_time': {
      const dwell = Number(inputValue || 0)
      if (dwell <= 2) return { factor: 'No current boarding delays', display_value: 'No delay', ...scoreBand(80) }
      if (dwell <= 5) return { factor: 'Boarding is a bit slower than normal', display_value: 'Slight delay', ...scoreBand(56) }
      return { factor: 'Heavy boarding delays at stops', display_value: 'Heavy delay', ...scoreBand(32) }
    }
    case 'cumulative_leg_time': {
      const leg = Number(inputValue || 0)
      if (leg <= 2) return { factor: 'Traffic is moving smoothly', display_value: 'No delay', ...scoreBand(78) }
      if (leg <= 5) return { factor: 'Traffic detected on route', display_value: 'Slight delay', ...scoreBand(55) }
      return { factor: 'Heavy traffic along the route', display_value: 'Heavy delay', ...scoreBand(36) }
    }
    case 'cumulative_stops': {
      const stops = Number(inputValue || 0)
      if (stops <= 5) return { factor: 'Short route segment ahead', display_value: 'Few stops', ...scoreBand(80) }
      if (stops <= 12) return { factor: 'Moderate number of stops ahead', display_value: 'Several stops', ...scoreBand(56) }
      return { factor: 'Many stops ahead may add delay', display_value: 'Many stops', ...scoreBand(35) }
    }
    case 'hour_of_day': {
      const hour = Number(inputValue || 0)
      const isPeak = (hour >= 7 && hour <= 9) || (hour >= 16 && hour <= 18)
      const displayValue = `${String(Math.round(hour)).padStart(2, '0')}:00`
      return { factor: 'Hour of day', display_value: displayValue, ...scoreBand(isPeak ? 42 : 72) }
    }
    case 'day_of_week': {
      const day = Number(inputValue || 0)
      const displayValue = day === 0 ? 'Mon' : day === 6 ? 'Sun' : 'Weekday'
      if (day === 0) return { factor: 'Day of week', display_value: displayValue, ...scoreBand(45) }
      if (day === 6) return { factor: 'Day of week', display_value: displayValue, ...scoreBand(76) }
      return { factor: 'Day of week', display_value: displayValue, ...scoreBand(62) }
    }
    case 'is_sunday': {
      const sunday = Number(inputValue || 0) === 1
      return { factor: 'Sunday service pattern', display_value: sunday ? 'Sun' : 'No', ...scoreBand(sunday ? 78 : 58) }
    }
    case 'mode': {
      const mode = String(inputValue || 'bus')
      return { factor: 'Transit mode', display_value: mode, ...scoreBand(64) }
    }
    case 'section_id': {
      return { factor: 'Route section signature', display_value: `#${inputValue}`, ...scoreBand(60) }
    }
    default:
      return { factor: inputKey, display_value: String(inputValue), ...scoreBand(58) }
  }
}

function buildModelDrivenFactors(forecastData) {
  const modelUsed = forecastData?.magi?.model_used
  const modelInputs = forecastData?.magi?.all_models?.[modelUsed]?.inputs
  if (!modelUsed || !modelInputs || typeof modelInputs !== 'object') {
    return []
  }

  return Object.entries(modelInputs)
    .filter(([, value]) => value !== null && value !== undefined)
    .map(([key, value]) => buildModelConsideration(key, value))
    .slice(0, 3)
}

function normalizeScoreFromForecast(data) {
  const winnerScore = data?.magi?.selection_context?.winner_score_100
  if (typeof winnerScore === 'number' && Number.isFinite(winnerScore)) {
    return Math.max(0, Math.min(100, winnerScore))
  }

  const predicted = data?.current?.predicted
  if (typeof predicted === 'number' && Number.isFinite(predicted) && predicted >= 0 && predicted <= 100) {
    return predicted
  }

  const label = String(data?.current?.label || '').toLowerCase()
  if (label === 'on_time') {
    return 90
  }
  if (label === 'minor') {
    return 68
  }
  if (label === 'moderate') {
    return 42
  }
  if (label === 'severe') {
    return 20
  }

  return null
}

function normalizeStopsPayload(data) {
  return Array.isArray(data?.stops) ? data.stops : []
}

function normalizeBikeStationsPayload(data) {
  if (!Array.isArray(data?.stations)) {
    return []
  }

  return data.stations.filter(
    (bike) => typeof bike?.lat === 'number' && Number.isFinite(bike.lat) && typeof bike?.lon === 'number' && Number.isFinite(bike.lon),
  )
}

function stationIcon(stop, isActive, opacity, outlineColor, isScoreLoading) {
  const modeClass = modeStyle(stop.mode)
  const fillClass =
    modeClass === 'railway'
      ? stop.is_parent_station === true
        ? 'is-parent'
        : 'is-child'
      : modeClass === 'light-rail'
        ? 'is-light-rail'
        : modeClass === 'bike'
          ? 'is-bike'
          : 'is-bus'
  const scoreClass = isScoreLoading ? 'is-score-loading' : ''

  if (modeClass === 'bike') {
    return divIcon({
      className: 'station-marker station-marker--bike',
      html: `<div class="station-shape station-shape--bike ${fillClass} ${scoreClass} ${isActive ? 'is-active' : ''}" style="opacity:${opacity.toFixed(3)}"><svg viewBox="0 0 26 26" width="26" height="26" aria-hidden="true"><path d="M13 4.25 L22.25 21.25 L3.75 21.25 Z" fill="#ffffff" stroke="${outlineColor}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/></svg></div>`,
      iconSize: [26, 26],
      iconAnchor: [13, 13],
      popupAnchor: [0, -11],
    })
  }

  return divIcon({
    className: 'station-marker',
    html: `<div class="station-shape station-shape--${modeClass} ${fillClass} ${scoreClass} ${isActive ? 'is-active' : ''}" style="opacity:${opacity.toFixed(3)};--marker-outline:${outlineColor};border-color:${outlineColor}"></div>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
    popupAnchor: [0, -10],
  })
}

function FocusMapOnStop({ stop }) {
  const map = useMap()

  useEffect(() => {
    if (!stop || typeof stop.lat !== 'number' || typeof stop.lon !== 'number') {
      return
    }
    map.flyTo([stop.lat, stop.lon], 14, { duration: 0.45 })
  }, [map, stop])

  return null
}

function MapViewTracker({ onViewChange }) {
  const updateView = (map) => {
    onViewChange({
      zoom: map.getZoom(),
      center: map.getCenter(),
      bounds: map.getBounds(),
    })
  }

  useMapEvents({
    move: (event) => updateView(event.target),
    zoom: (event) => updateView(event.target),
    moveend: (event) => updateView(event.target),
    zoomend: (event) => updateView(event.target),
  })

  return null
}

function LandingMap({ selectedStop, onSelectStop, onEnterForecast, darkMode, onToggleDarkMode }) {
  const [query, setQuery] = useState('')
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [modeMenuOpen, setModeMenuOpen] = useState(false)
  const [selectedModes, setSelectedModes] = useState(['railway', 'bus'])
  const [stops, setStops] = useState({ loading: true, error: '', items: [] })
  const [results, setResults] = useState({ loading: false, error: '', items: [] })
  const [bikeCatalog, setBikeCatalog] = useState({ loading: true, error: '', items: [] })
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [lastRefreshAt, setLastRefreshAt] = useState(null)
  const [mapView, setMapView] = useState({ zoom: 11, center: null, bounds: null })
  const [scoreByStopId, setScoreByStopId] = useState({})
  const searchRef = useRef(null)
  const modeFilterRef = useRef(null)
  const pendingScoreRequestsRef = useRef(new Set())
  const scoreRetryAfterRef = useRef(new Map())
  const scoreByStopIdRef = useRef({})

  useEffect(() => {
    scoreByStopIdRef.current = scoreByStopId
  }, [scoreByStopId])

  const selectedModeSet = useMemo(() => new Set(selectedModes), [selectedModes])

  function toggleMode(nextMode) {
    setSelectedModes((prev) => {
      if (prev.includes(nextMode)) {
        return prev.filter((mode) => mode !== nextMode)
      }
      return [...prev, nextMode]
    })
  }

  useEffect(() => {
    let cancelled = false

    async function loadStops() {
      setStops((prev) => ({ ...prev, loading: true, error: '' }))
      try {
        const data = await fetchJson('/stops?limit=5000')
        if (!cancelled) {
          setStops({ loading: false, error: '', items: normalizeStopsPayload(data) })
        }
      } catch (error) {
        if (!cancelled) {
          setStops({
            loading: false,
            error: describeFetchError(error, '/stops'),
            items: [],
          })
        }
      }
    }

    loadStops()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!dropdownOpen) {
      return undefined
    }

    function handleClickOutside(event) {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setDropdownOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [dropdownOpen])

  useEffect(() => {
    if (!modeMenuOpen) {
      return undefined
    }

    function handleClickOutside(event) {
      if (modeFilterRef.current && !modeFilterRef.current.contains(event.target)) {
        setModeMenuOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [modeMenuOpen])

  useEffect(() => {
    let cancelled = false

    async function runSearch() {
      const trimmed = query.trim()
      if (!trimmed) {
        setResults({ loading: false, error: '', items: [] })
        return
      }

      setResults((prev) => ({ ...prev, loading: true, error: '' }))

      try {
        const data = await fetchJson(`/stops/search?q=${encodeURIComponent(trimmed)}&limit=8`)
        if (!cancelled) {
          setResults({
            loading: false,
            error: '',
            items: Array.isArray(data?.stops) ? data.stops : [],
          })
        }
      } catch (error) {
        if (!cancelled) {
          setResults({
            loading: false,
            error: describeFetchError(error, '/stops/search'),
            items: [],
          })
        }
      }
    }

    runSearch()
    return () => {
      cancelled = true
    }
  }, [query])

  useEffect(() => {
    let cancelled = false

    async function loadBikeCatalog() {
      setBikeCatalog((prev) => ({ ...prev, loading: true, error: '' }))
      try {
        const data = await fetchJson('/bikeshare/stations?limit=5000')
        if (cancelled) {
          return
        }

        setBikeCatalog({ loading: false, error: '', items: normalizeBikeStationsPayload(data) })
      } catch (error) {
        if (!cancelled) {
          setBikeCatalog({ loading: false, error: describeFetchError(error, '/bikeshare/stations'), items: [] })
        }
      }
    }

    loadBikeCatalog()
    return () => {
      cancelled = true
    }
  }, [])

  const visibleStops = useMemo(() => {
    if (mapView.zoom < MAP_RENDER_FADE_START_ZOOM || !mapView.bounds) {
      return []
    }

    const modeFiltered = stops.items.filter((stop) => selectedModeSet.has(normalizeTransitFilterMode(stop.mode)))

    return modeFiltered.filter((stop) => mapView.bounds.contains([stop.lat, stop.lon]))
  }, [mapView.bounds, mapView.zoom, selectedModeSet, stops.items])

  const visibleBikeStations = useMemo(() => {
    if (!mapView.bounds) {
      return []
    }
    if (!selectedModeSet.has('bike')) {
      return []
    }

    return bikeCatalog.items.filter((bike) => mapView.bounds.contains([bike.lat, bike.lon]))
  }, [bikeCatalog.items, mapView.bounds, selectedModeSet])

  useEffect(() => {
    const batchSize = 80
    const maxConcurrent = 8
    const nowMs = Date.now()
    const viewCenter = mapView.center

    const missingStops = visibleStops
      .filter((stop) => !Object.prototype.hasOwnProperty.call(scoreByStopIdRef.current, stop.stop_id))
      .filter((stop) => !pendingScoreRequestsRef.current.has(stop.stop_id))
      .filter((stop) => {
        const retryAfter = scoreRetryAfterRef.current.get(stop.stop_id) || 0
        return retryAfter <= nowMs
      })
      .slice(0, batchSize)

    if (missingStops.length === 0) {
      return undefined
    }

    const prioritizedStops = [...missingStops].sort((a, b) => {
      if (!viewCenter) {
        return 0
      }
      const aDist = (a.lat - viewCenter.lat) ** 2 + (a.lon - viewCenter.lng) ** 2
      const bDist = (b.lat - viewCenter.lat) ** 2 + (b.lon - viewCenter.lng) ** 2
      return aDist - bDist
    })

    let cancelled = false
    let nextIndex = 0
    let activeCount = 0

    function launchNext() {
      if (cancelled) {
        return
      }

      while (activeCount < maxConcurrent && nextIndex < prioritizedStops.length) {
        const stop = prioritizedStops[nextIndex]
        nextIndex += 1
        activeCount += 1
        pendingScoreRequestsRef.current.add(stop.stop_id)

        fetchJson(`/station/${encodeURIComponent(stop.stop_id)}`)
          .then((data) => {
            if (cancelled) {
              return
            }

            const nextScore = normalizeScoreFromForecast(data)
            scoreRetryAfterRef.current.delete(stop.stop_id)
            setScoreByStopId((prev) => {
              if (Object.prototype.hasOwnProperty.call(prev, stop.stop_id)) {
                return prev
              }
              return { ...prev, [stop.stop_id]: nextScore }
            })
          })
          .catch(() => {
            scoreRetryAfterRef.current.set(stop.stop_id, Date.now() + 30000)
          })
          .finally(() => {
            pendingScoreRequestsRef.current.delete(stop.stop_id)
            activeCount -= 1
            launchNext()
          })
      }
    }

    launchNext()
    return () => {
      cancelled = true
    }
  }, [mapView.center, visibleStops])

  async function refreshMapData() {
    if (isRefreshing) {
      return
    }

    setIsRefreshing(true)
    setStops((prev) => ({ ...prev, loading: true, error: '' }))
    setBikeCatalog((prev) => ({ ...prev, loading: true, error: '' }))
    pendingScoreRequestsRef.current.clear()
    scoreRetryAfterRef.current.clear()
    setScoreByStopId({})

    try {
      const [stopsResult, bikesResult] = await Promise.allSettled([fetchJson('/stops?limit=5000'), fetchJson('/bikeshare/stations?limit=5000')])

      if (stopsResult.status === 'fulfilled') {
        setStops({
          loading: false,
          error: '',
          items: normalizeStopsPayload(stopsResult.value),
        })
      } else {
        setStops((prev) => ({
          loading: false,
          error: describeFetchError(stopsResult.reason, '/stops'),
          items: prev.items,
        }))
      }

      if (bikesResult.status === 'fulfilled') {
        setBikeCatalog({ loading: false, error: '', items: normalizeBikeStationsPayload(bikesResult.value) })
      } else {
        setBikeCatalog((prev) => ({
          loading: false,
          error: describeFetchError(bikesResult.reason, '/bikeshare/stations'),
          items: prev.items,
        }))
      }

      setLastRefreshAt(new Date())
    } finally {
      setIsRefreshing(false)
    }
  }

  return (
    <main className="landing-shell">
      <section className="landing-top">
        <div className="landing-top-head">
          <div>
            <p className="landing-eyebrow">ForeTransit</p>
            <h1 className="landing-title">TTC Station Map</h1>
          </div>
          <button type="button" className="theme-toggle-btn" onClick={onToggleDarkMode}>
            {darkMode ? 'Light mode' : 'Dark mode'}
          </button>
        </div>

        <div className="map-search-panel" ref={searchRef}>
          <input
            className="search-input"
            type="text"
            value={query}
            onChange={(e) => {
              const next = e.target.value
              setQuery(next)
              setDropdownOpen(!!next.trim())
            }}
            onFocus={() => setDropdownOpen(!!query.trim())}
            placeholder="Search station names"
          />

          {dropdownOpen && query.trim() ? (
            <div className="search-dropdown">
              {results.loading ? <span className="search-note">Searching...</span> : null}
              {results.error ? <span className="search-error">{results.error}</span> : null}
              {!results.loading && !results.error && results.items.length === 0 ? (
                <span className="search-note">No station matches found</span>
              ) : null}
              {results.items.map((stop) => (
                <button
                  key={stop.stop_id}
                  type="button"
                  className="search-item"
                  onClick={() => {
                    onSelectStop(stop)
                    setQuery(stop.stop_name)
                    setDropdownOpen(false)
                  }}
                >
                  <strong>{stop.stop_name}</strong>
                  <span className={`station-tag station-tag--${modeStyle(stop.mode)}`}>{transitTagLabel(stop.mode)}</span>
                </button>
              ))}
            </div>
          ) : null}
        </div>

        <div className="map-actions">
          <button type="button" className="map-action-btn open-forecast-btn" onClick={onEnterForecast} disabled={!selectedStop}>
            {selectedStop ? `Open forecast for ${selectedStop.stop_name}` : 'Select a station'}
          </button>
          <button type="button" className="map-action-btn refresh-data-btn" onClick={refreshMapData} disabled={isRefreshing}>
            {isRefreshing ? 'Refreshing...' : 'Refresh data'}
          </button>
          <div className="mode-filter-dropdown" ref={modeFilterRef}>
            <button
              type="button"
              className={`mode-filter-trigger ${modeMenuOpen ? 'is-open' : ''}`}
              onClick={() => setModeMenuOpen((open) => !open)}
              aria-expanded={modeMenuOpen}
              aria-haspopup="menu"
            >
              Transit types ({selectedModes.length})
            </button>
            {modeMenuOpen ? (
              <div className="mode-filter-menu" role="group" aria-label="Filter map station types">
                {TRANSIT_MODE_OPTIONS.map((option) => (
                  <label key={option.value} className="mode-filter-option">
                    <input
                      type="checkbox"
                      checked={selectedModes.includes(option.value)}
                      onChange={() => toggleMode(option.value)}
                    />
                    <span>{option.label}</span>
                  </label>
                ))}
              </div>
            ) : null}
          </div>
        </div>
        {stops.error ? <p className="search-error">{stops.error}</p> : null}
        {bikeCatalog.error ? <p className="search-error">{bikeCatalog.error}</p> : null}
        {lastRefreshAt ? (
          <p className="refresh-note">Last refreshed {lastRefreshAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</p>
        ) : null}
      </section>

      <section className="map-box">
        <MapContainer center={[43.6532, -79.3832]} zoom={11} minZoom={10} maxZoom={17} className="ttc-map">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <FocusMapOnStop stop={selectedStop} />
          <MapViewTracker onViewChange={setMapView} />

          {visibleStops.map((stop) => {
            const isActive = selectedStop?.stop_id === stop.stop_id
            const revealOpacity = zoomRevealOpacity(mapView.zoom)
            const opacity = isActive ? 1 : edgeFadeOpacity(stop, mapView.bounds) * revealOpacity
            const hasScore = Object.prototype.hasOwnProperty.call(scoreByStopId, stop.stop_id)
            const score = scoreByStopId[stop.stop_id]
            const outlineColor = colorByScore(score)
            return (
              <Marker
                key={stop.stop_id}
                position={[stop.lat, stop.lon]}
                icon={stationIcon(stop, isActive, opacity, outlineColor, !hasScore)}
                eventHandlers={{
                  click: () => onSelectStop(stop),
                }}
              >
                <Popup>
                  <div className="popup-card">
                    <strong>{stop.stop_name}</strong>
                    <p>
                      <span className={`station-tag station-tag--${modeStyle(stop.mode)}`}>{transitTagLabel(stop.mode)}</span>
                    </p>
                    <button type="button" onClick={onEnterForecast}>
                      Open forecast
                    </button>
                  </div>
                </Popup>
              </Marker>
            )
          })}

          {visibleBikeStations.map((bike) => (
            <Marker
              key={`bike-${bike.station_id}`}
              position={[bike.lat, bike.lon]}
              icon={stationIcon(
                { mode: 'bike', is_parent_station: false },
                false,
                0.95,
                colorByBikeAvailability(bike.bikes_available),
                bikeCatalog.loading,
              )}
            >
              <Popup>
                <div className="popup-card">
                  <strong>{bike.name}</strong>
                  <p>Bike share</p>
                  <p>Bikes: {bike.bikes_available} · Docks: {bike.docks_available}</p>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>

        <aside className="map-legend" aria-label="Map legend">
          <p className="map-legend-title">Legend</p>
          <div className="map-legend-grid">
            <div className="legend-item">
              <span className="legend-shape legend-shape--railway" />
              <span>Train</span>
            </div>
            <div className="legend-item">
              <span className="legend-shape legend-shape--bus" />
              <span>Bus</span>
            </div>
            <div className="legend-item">
              <span className="legend-shape legend-shape--bike" aria-hidden="true">
                <svg viewBox="0 0 16 16" width="14" height="14">
                  <path d="M8 2 L14 13 L2 13 Z" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
                </svg>
              </span>
              <span>Bikeshare</span>
            </div>
          </div>

          <p className="map-legend-subtitle">Outline score</p>
          <div className="map-legend-grid">
            <div className="legend-item">
              <span className="legend-dot legend-dot--good" />
              <span>Good</span>
            </div>
            <div className="legend-item">
              <span className="legend-dot legend-dot--fair" />
              <span>Moderate</span>
            </div>
            <div className="legend-item">
              <span className="legend-dot legend-dot--poor" />
              <span>Poor</span>
            </div>
          </div>
        </aside>
      </section>

      <p className="map-render-hint">
        {mapView.zoom < MAP_RENDER_FADE_START_ZOOM
          ? 'Zoom in to start revealing likely-use stops.'
          : `Rendering ${visibleStops.length} transit stops and ${visibleBikeStations.length} bikeshare stations in the current view.`}
      </p>
    </main>
  )
}

function ForecastDashboard({ selectedStop, onBackToMap }) {
  const [query, setQuery] = useState('')
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [results, setResults] = useState({ loading: false, error: '', items: [] })
  const [forecast, setForecast] = useState({ loading: false, error: '', data: null })
  const [now, setNow] = useState(() => new Date())
  const [liveWeather, setLiveWeather] = useState({ loading: true, error: '', data: null })
  const searchRef = useRef(null)

  useEffect(() => {
    if (selectedStop?.stop_name) {
      setQuery(selectedStop.stop_name)
    }
    if (selectedStop?.stop_id) {
      selectStop(selectedStop)
    }
  }, [selectedStop?.stop_id])

  function backToMap() {
    onBackToMap()
  }

  useEffect(() => {
    const timer = window.setInterval(() => {
      setNow(new Date())
    }, 1000)

    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    let cancelled = false

    async function fetchWeather() {
      setLiveWeather((prev) => ({ ...prev, loading: true, error: '' }))
      try {
        const data = await fetchJson('/weather/current')
        if (!cancelled) {
          if (data?.error) {
            setLiveWeather({ loading: false, error: data.error, data: null })
            return
          }
          setLiveWeather({ loading: false, error: '', data })
        }
      } catch (error) {
        if (!cancelled) {
          setLiveWeather({
            loading: false,
            error: describeFetchError(error, '/weather/current'),
            data: null,
          })
        }
      }
    }

    fetchWeather()
    const weatherTimer = window.setInterval(fetchWeather, 60000)
    return () => {
      cancelled = true
      window.clearInterval(weatherTimer)
    }
  }, [])

  useEffect(() => {
    if (!dropdownOpen) {
      return undefined
    }

    function handleClickOutside(event) {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setDropdownOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [dropdownOpen])

  useEffect(() => {
    let cancelled = false

    async function runSearch() {
      const trimmed = query.trim()
      if (!trimmed) {
        setResults({ loading: false, error: '', items: [] })
        return
      }

      setResults((prev) => ({ ...prev, loading: true, error: '' }))

      try {
        const data = await fetchJson(`/stops/search?q=${encodeURIComponent(trimmed)}&limit=8`)
        if (!cancelled) {
          setResults({
            loading: false,
            error: '',
            items: Array.isArray(data?.stops) ? data.stops : [],
          })
        }
      } catch (error) {
        if (!cancelled) {
          setResults({
            loading: false,
            error: describeFetchError(error, '/stops/search'),
            items: [],
          })
        }
      }
    }

    runSearch()
    return () => {
      cancelled = true
    }
  }, [query])

  async function selectStop(stop) {
    setQuery(stop.stop_name)
    setDropdownOpen(false)
    setForecast({ loading: true, error: '', data: null })

    try {
      const data = await fetchJson(`/station/${encodeURIComponent(stop.stop_id)}`)
      if (data?.error) {
        throw new Error(data.error)
      }
      setForecast({ loading: false, error: '', data })
    } catch (error) {
      setForecast({
        loading: false,
        error: describeFetchError(error, '/station/{stop_id}'),
        data: null,
      })
    }
  }

  const magi = forecast.data?.magi
  const winnerScore = magi?.selection_context?.winner_score_100
  const modelDrivenFactors = buildModelDrivenFactors(forecast.data)
  const fallbackFactors = Array.isArray(forecast.data?.top_factors) ? forecast.data.top_factors : []
  const factors = modelDrivenFactors.length > 0 ? modelDrivenFactors : fallbackFactors
  const usingModelFactors = modelDrivenFactors.length > 0
  const scoreHeading = 'Reliability Score'
  const displayScore = typeof winnerScore === 'number' ? Math.round(winnerScore) : '--'
  const scoreColor = colorByScore(winnerScore)
  const serviceStatus = magi?.service_status || serviceStatusFromScore(winnerScore)
  const nextArrival = forecast.data?.current?.estimated_arrival || 'loading'
  const nextArrivalMins = forecast.data?.current?.estimated_arrival_in_min
  const etaSuffix = typeof nextArrivalMins === 'number' ? ` (${nextArrivalMins} min)` : ''
  const adviceText = magi?.advice_text || adviceFromScore(winnerScore)
  const displayTime = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })
  const displayDate = now.toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', year: 'numeric' }).replace(/\//g, ' / ')
  const displayWeekday = now.toLocaleDateString('en-US', { weekday: 'long' }).toLowerCase().split('').join(' ')
  const liveTemp = liveWeather.data?.temperature_c
  const liveRain = liveWeather.data?.rain_mm
  const liveWind = liveWeather.data?.wind_kmh
  const liveTempF = typeof liveTemp === 'number' ? Math.round((liveTemp * 9) / 5 + 32) : null
  const liveWindMph = typeof liveWind === 'number' ? Math.round(liveWind * 0.621371) : null
  const nextVehicleLine = typeof nextArrivalMins === 'number' ? `${nextArrivalMins} MINUTES` : String(nextArrival || 'loading').toUpperCase()

  return (
    <main className="minimal-shell">
      <header className="info-booth" aria-live="polite">
        <div className="booth-left">
          <button type="button" className="back-map-btn" onClick={backToMap}>
            {'<<< back to map'}
          </button>
        </div>

        <div className="booth-right">
          <p className="booth-brand">foretransit</p>
          <p className="booth-date">{displayDate}</p>
          <p className="booth-weekday">{displayWeekday}</p>
        </div>
      </header>

      <p className="forecast-time">{displayTime}</p>

      <section className="search-panel" ref={searchRef}>
        <input
          className="search-input"
          type="text"
          value={query}
          onChange={(e) => {
            const next = e.target.value
            setQuery(next)
            setDropdownOpen(!!next.trim())
          }}
          onFocus={() => setDropdownOpen(!!query.trim())}
          placeholder=">>> search station names"
        />

        {dropdownOpen && query.trim() ? (
          <div className="search-dropdown">
            {results.loading ? <span className="search-note">Searching...</span> : null}
            {results.error ? <span className="search-error">{results.error}</span> : null}
            {!results.loading && !results.error && results.items.length === 0 ? (
              <span className="search-note">No station matches found</span>
            ) : null}
            {results.items.map((stop) => (
              <button
                key={stop.stop_id}
                type="button"
                className="search-item"
                onClick={() => selectStop(stop)}
              >
                <strong>{stop.stop_name}</strong>
                <span>{stop.mode}</span>
              </button>
            ))}
          </div>
        ) : null}
      </section>

      <section className="result-panel">
        {forecast.error ? <p className="search-error">{forecast.error}</p> : null}

        <div className="weather-widget">
          <div className="widget-top-row">
            <div className="widget-left">
              <div className="widget-score-col">
                <p className="widget-station">transit reliability...</p>
                <div className="widget-score-row">
                  <p className="widget-score" style={{ backgroundColor: scoreColor }}>{displayScore}</p>
                </div>
                <p className="widget-service-tag">{String(serviceStatus || 'Unknown').toUpperCase()}</p>
                <p className="widget-advice-line">&quot;{String(adviceText || 'loading.').toLowerCase()}&quot;</p>
              </div>

              <div className="widget-meta-col">
                <div className="widget-meta-group">
                  <p className="widget-meta-label">confidence...</p>
                  <p className="widget-meta-value">{compactConfidence(winnerScore)}</p>
                </div>
                <div className="widget-meta-group">
                  <p className="widget-meta-label">service status...</p>
                  <p className="widget-meta-value">{String(serviceStatus || 'Unknown').toUpperCase()}</p>
                </div>
                <div className="widget-meta-group">
                  <p className="widget-meta-label">next vehicle (est. arrival)</p>
                  <p className="widget-meta-value">{nextVehicleLine}</p>
                </div>
              </div>

              <div className="score-band-legend" aria-hidden="true">
                <span><i className="legend-dot legend-dot--bad" />0-49</span>
                <span><i className="legend-dot legend-dot--warn" />50-89</span>
                <span><i className="legend-dot legend-dot--good" />90-100</span>
              </div>
            </div>

            <div className="widget-summary-box">
              <p className="widget-summary-title">temperature</p>
              <p className="widget-arrival">
                {typeof liveTemp === 'number' ? `${Math.round(liveTemp)}°C` : 'loading'}
                {' // '}
                {typeof liveTempF === 'number' ? `${liveTempF}°F` : 'loading'}
              </p>

              <p className="widget-summary-title">rain...</p>
              <p className="widget-arrival">{typeof liveRain === 'number' ? `${liveRain}mm` : 'loading'}</p>

              <p className="widget-summary-title">wind...</p>
              <p className="widget-arrival">
                {typeof liveWind === 'number' ? `${liveWind} kmh` : 'loading'}
                {' // '}
                {typeof liveWindMph === 'number' ? `${liveWindMph} mph` : 'loading'}
              </p>
            </div>
          </div>

          <div className="widget-right">
            <p className="widget-title">live trip status...</p>
            {factors.length > 0 ? (
              <ul className="factor-list">
                {factors.slice(0, 3).map((factor, idx) => {
                  const rowTitle = factorRowTitle(factor)
                  const rowBadge = factorBadgeLabel(factor)
                  return (
                    <li key={`${factor.factor}-${idx}`} className={`factor-item ${factorToneClass(factor)}`}>
                      <span className={`factor-text${rowTitle === 'BOARDING DELAYS' ? ' factor-text--boarding' : ''}`}>
                        {rowTitle}
                      </span>
                      <span className={`factor-score${rowBadge === 'no delays' ? ' factor-score--no-delays' : ''}`}>
                        {rowBadge}
                      </span>
                    </li>
                  )
                })}
              </ul>
            ) : (
              <p className="widget-empty">
                {forecast.loading
                  ? 'Loading live trip status...'
                  : 'No significant live impacts detected right now.'}
              </p>
            )}
          </div>
        </div>
      </section>
    </main>
  )
}

function App() {
  const [view, setView] = useState('map')
  const [selectedStop, setSelectedStop] = useState(null)
  const [darkMode, setDarkMode] = useState(() => {
    try {
      return window.localStorage.getItem('foretransit-theme') === 'dark'
    } catch {
      return false
    }
  })

  useEffect(() => {
    document.title = view === 'forecast' ? 'ForeTransit | Forecast Interface' : 'ForeTransit | Map Interface'
  }, [view])

  useEffect(() => {
    document.body.classList.toggle('theme-dark', darkMode)
    try {
      window.localStorage.setItem('foretransit-theme', darkMode ? 'dark' : 'light')
    } catch {
      // Ignore storage write issues and keep in-memory theme state.
    }
  }, [darkMode])

  if (view === 'forecast') {
    return <ForecastDashboard selectedStop={selectedStop} onBackToMap={() => setView('map')} />
  }

  return (
    <LandingMap
      selectedStop={selectedStop}
      onSelectStop={setSelectedStop}
      onEnterForecast={() => setView('forecast')}
      darkMode={darkMode}
      onToggleDarkMode={() => setDarkMode((prev) => !prev)}
    />
  )
}

export default App
