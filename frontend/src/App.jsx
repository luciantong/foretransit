import { useEffect, useMemo, useRef, useState } from 'react'
import { MapContainer, Marker, Popup, TileLayer, useMap, useMapEvents } from 'react-leaflet'
import { divIcon } from 'leaflet'
import './App.css'

const API_ROOT = '/api'
const MAP_RENDER_START_ZOOM = 14

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
  return 'bus'
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

function stationIcon(stop, isActive, opacity) {
  const modeClass = modeStyle(stop.mode)
  const fillClass =
    modeClass === 'railway'
      ? stop.is_parent_station === true
        ? 'is-parent'
        : 'is-child'
      : 'is-bus'

  return divIcon({
    className: 'station-marker',
    html: `<div class="station-shape station-shape--${modeClass} ${fillClass} ${isActive ? 'is-active' : ''}" style="opacity:${opacity.toFixed(3)}"></div>`,
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
  useMapEvents({
    moveend: (event) => {
      const map = event.target
      onViewChange({
        zoom: map.getZoom(),
        center: map.getCenter(),
        bounds: map.getBounds(),
      })
    },
    zoomend: (event) => {
      const map = event.target
      onViewChange({
        zoom: map.getZoom(),
        center: map.getCenter(),
        bounds: map.getBounds(),
      })
    },
  })

  return null
}

function LandingMap({ selectedStop, onSelectStop, onEnterForecast }) {
  const [query, setQuery] = useState('')
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [stops, setStops] = useState({ loading: true, error: '', items: [] })
  const [results, setResults] = useState({ loading: false, error: '', items: [] })
  const [mapView, setMapView] = useState({ zoom: 11, center: null, bounds: null })
  const searchRef = useRef(null)

  useEffect(() => {
    let cancelled = false

    async function loadStops() {
      setStops((prev) => ({ ...prev, loading: true, error: '' }))
      try {
        const res = await fetch(`${API_ROOT}/stops?limit=2500`)
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const data = await res.json()
        if (!cancelled) {
          setStops({ loading: false, error: '', items: Array.isArray(data?.stops) ? data.stops : [] })
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
    let cancelled = false

    async function runSearch() {
      const trimmed = query.trim()
      if (!trimmed) {
        setResults({ loading: false, error: '', items: [] })
        return
      }

      setResults((prev) => ({ ...prev, loading: true, error: '' }))

      try {
        const res = await fetch(`${API_ROOT}/stops/search?q=${encodeURIComponent(trimmed)}&limit=8`)
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const data = await res.json()
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

  const visibleStops = useMemo(() => {
    if (mapView.zoom < MAP_RENDER_START_ZOOM || !mapView.bounds) {
      return []
    }

    const byBounds = stops.items.filter((stop) => mapView.bounds.contains([stop.lat, stop.lon]))

    // Keep higher-intent rail parent stations; platform children add visual noise.
    const likelyStops = byBounds.filter((stop) => {
      if (stop.mode === 'railway') {
        return stop.is_parent_station === true
      }
      return true
    })

    // For non-rail modes, de-duplicate by name to reduce directional stop clutter.
    const seenNames = new Set()
    const deduped = []
    for (const stop of likelyStops) {
      if (stop.mode === 'railway') {
        deduped.push(stop)
        continue
      }
      const key = String(stop.stop_name || '').toLowerCase().trim()
      if (!key || seenNames.has(key)) {
        continue
      }
      seenNames.add(key)
      deduped.push(stop)
    }

    return deduped
  }, [mapView.bounds, mapView.zoom, stops.items])

  return (
    <main className="landing-shell">
      <section className="landing-top">
        <p className="landing-eyebrow">ForeTransit</p>
        <h1 className="landing-title">TTC Station Map</h1>

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
                  <span>{stop.mode}</span>
                </button>
              ))}
            </div>
          ) : null}
        </div>

        <div className="map-actions">
          <button type="button" className="open-forecast-btn" onClick={onEnterForecast} disabled={!selectedStop}>
            {selectedStop ? `Open forecast for ${selectedStop.stop_name}` : 'Select a station'}
          </button>
        </div>
        {stops.error ? <p className="search-error">{stops.error}</p> : null}
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
            const opacity = isActive ? 1 : edgeFadeOpacity(stop, mapView.bounds)
            return (
              <Marker
                key={stop.stop_id}
                position={[stop.lat, stop.lon]}
                icon={stationIcon(stop, isActive, opacity)}
                eventHandlers={{
                  click: () => onSelectStop(stop),
                }}
              >
                <Popup>
                  <div className="popup-card">
                    <strong>{stop.stop_name}</strong>
                    <p>{stop.mode}</p>
                    <button type="button" onClick={onEnterForecast}>
                      Open forecast
                    </button>
                  </div>
                </Popup>
              </Marker>
            )
          })}
        </MapContainer>
      </section>

      <p className="map-render-hint">
        {mapView.zoom < MAP_RENDER_START_ZOOM
          ? 'Zoom in to 50%+ to render likely-use stops.'
          : `Rendering ${visibleStops.length} likely-use stops in the current view.`}
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
        const res = await fetch(`${API_ROOT}/weather/current`)
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const data = await res.json()
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
        const res = await fetch(`${API_ROOT}/stops/search?q=${encodeURIComponent(trimmed)}&limit=8`)
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const data = await res.json()
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
      const res = await fetch(`${API_ROOT}/station/${encodeURIComponent(stop.stop_id)}`)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const data = await res.json()
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
  const factors = Array.isArray(forecast.data?.top_factors) ? forecast.data.top_factors : []
  const scoreHeading = 'Predicted standardized score'
  const displayScore = typeof winnerScore === 'number' ? Math.round(winnerScore) : '--'
  const displayTime = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  const displayDate = now.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' })
  const liveTemp = liveWeather.data?.temperature_c
  const liveRain = liveWeather.data?.rain_mm
  const liveWind = liveWeather.data?.wind_kmh

  return (
    <main className="minimal-shell">
      <header className="info-booth" aria-live="polite">
        <div className="booth-left">
          <button type="button" className="back-map-btn" onClick={backToMap}>
            Back to map
          </button>
        </div>

        <div className="booth-center">
          <p className="booth-label">Weather</p>
          {liveWeather.loading ? <p className="booth-value">Updating weather...</p> : null}
          {liveWeather.error ? <p className="booth-value error">{liveWeather.error}</p> : null}
          {!liveWeather.loading && !liveWeather.error ? (
            <p className="booth-value">
              Temp {typeof liveTemp === 'number' ? `${Math.round(liveTemp)} C` : 'N/A'}
              {' · '}Rain {typeof liveRain === 'number' ? `${liveRain} mm` : 'N/A'}
              {' · '}Wind {typeof liveWind === 'number' ? `${liveWind} km/h` : 'N/A'}
            </p>
          ) : null}
        </div>

        <div className="booth-right">
          <p className="booth-label">Time</p>
          <p className="booth-time">{displayTime}</p>
          <p className="booth-date">{displayDate}</p>
        </div>
      </header>

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
        {forecast.loading ? <p>Loading model selection...</p> : null}
        {forecast.error ? <p className="search-error">{forecast.error}</p> : null}

        <div className="weather-widget">
          <div className="widget-left">
            <p className="widget-station">{scoreHeading}</p>
            <div className="widget-score-row">
              <p className="widget-score">{displayScore}</p>
              <span className="widget-score-max">/100</span>
            </div>
            <p className="widget-model-line">
              Model selected: {formatModelName(magi?.model_used)}
            </p>
          </div>

          <div className="widget-right">
            <p className="widget-title">Confounding Factors</p>
            {factors.length > 0 ? (
              <ul className="factor-list">
                {factors.slice(0, 3).map((factor, idx) => (
                  <li key={`${factor.factor}-${idx}`}>
                    <span>{factor.factor}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="widget-empty">Confounding factors will appear after forecast data loads.</p>
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

  useEffect(() => {
    document.title = view === 'forecast' ? 'ForeTransit | Forecast Interface' : 'ForeTransit | Map Interface'
  }, [view])

  if (view === 'forecast') {
    return <ForecastDashboard selectedStop={selectedStop} onBackToMap={() => setView('map')} />
  }

  return (
    <LandingMap
      selectedStop={selectedStop}
      onSelectStop={setSelectedStop}
      onEnterForecast={() => setView('forecast')}
    />
  )
}

export default App
