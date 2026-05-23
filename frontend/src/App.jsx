import { useEffect, useMemo, useRef, useState } from 'react'
import { MapContainer, Marker, Popup, TileLayer, useMap, useMapEvents } from 'react-leaflet'
import { divIcon } from 'leaflet'
import './App.css'

const API_ROOT = 'http://127.0.0.1:8000'
const MAP_MIN_ZOOM = 10
const MAP_MAX_ZOOM = 17
const STATION_FADE_START_ZOOM = 13.5
const STATION_LOAD_ZOOM = 15
const MAX_RENDER_DISTANCE_KM = 1.8

function formatDelay(delayMin) {
  if (typeof delayMin !== 'number' || Number.isNaN(delayMin)) {
    return 'N/A'
  }
  if (delayMin <= 0) {
    return 'On time'
  }
  return `${delayMin.toFixed(1)} min late`
}

function statusClass(health) {
  return health?.ok ? 'ok' : 'down'
}

function ZoomTracker({ onViewChange }) {
  useMapEvents({
    zoomend: (event) => onViewChange(event.target.getZoom(), event.target.getCenter()),
    moveend: (event) => onViewChange(event.target.getZoom(), event.target.getCenter()),
  })

  return null
}

function FocusMap({ stop }) {
  const map = useMap()

  useEffect(() => {
    if (!stop || typeof stop.lat !== 'number' || typeof stop.lon !== 'number') {
      return
    }

    map.flyTo([stop.lat, stop.lon], Math.max(map.getZoom(), STATION_LOAD_ZOOM), {
      duration: 0.5,
    })
  }, [map, stop])

  return null
}

function haversineKm(lat1, lon1, lat2, lon2) {
  const radiusKm = 6371
  const dLat = ((lat2 - lat1) * Math.PI) / 180
  const dLon = ((lon2 - lon1) * Math.PI) / 180
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2
  return radiusKm * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

function stationIcon(mode, active, opacity) {
  return divIcon({
    className: 'station-marker',
    html: `<div class="station-shape station-shape--${mode} ${active ? 'is-active' : ''}" style="opacity:${opacity};"></div>`,
    iconSize: [18, 18],
    iconAnchor: [9, 9],
    popupAnchor: [0, -8],
  })
}

function LandingMap({ selectedStop, onSelectStop, onEnterForecast }) {
  const [query, setQuery] = useState('')
  const [dropdownOpen, setDropdownOpen] = useState(false)
    const searchRef = useRef(null)
    useEffect(() => {
      if (!dropdownOpen) return;
      function handleClick(e) {
        if (searchRef.current && !searchRef.current.contains(e.target)) {
          setDropdownOpen(false)
        }
      }
      document.addEventListener('mousedown', handleClick)
      return () => document.removeEventListener('mousedown', handleClick)
    }, [dropdownOpen])
  const [stops, setStops] = useState({ loading: true, error: '', items: [] })
  const [searchResults, setSearchResults] = useState({ loading: false, error: '', items: [] })
  const [mapZoom, setMapZoom] = useState(11)
  const [mapCenter, setMapCenter] = useState([43.6532, -79.3832])
  const [focusStop, setFocusStop] = useState(null)

  useEffect(() => {
    let cancelled = false

    async function loadStops() {
      setStops((prev) => ({ ...prev, loading: true, error: '' }))
      try {
        const q = encodeURIComponent(query.trim())
        const res = await fetch(`${API_ROOT}/stops?q=${q}&limit=5000`)
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const data = await res.json()
        const items = Array.isArray(data?.stops) ? data.stops : []
        if (!cancelled) {
          setStops({ loading: false, error: '', items })
        }
      } catch (error) {
        if (!cancelled) {
          setStops({
            loading: false,
            error: error instanceof Error ? error.message : 'Could not load subway stops',
            items: [],
          })
        }
      }
    }

    loadStops()
    setDropdownOpen(!!query.trim())
    return () => {
      cancelled = true
    }
  }, [query, mapZoom])

  useEffect(() => {
    let cancelled = false

    async function loadSearchResults() {
      const trimmed = query.trim()

      if (!trimmed) {
        setSearchResults({ loading: false, error: '', items: [] })
        return
      }

      setSearchResults((prev) => ({ ...prev, loading: true, error: '' }))

      try {
        const q = encodeURIComponent(trimmed)
        const res = await fetch(`${API_ROOT}/stops/search?q=${q}&limit=8`)
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const data = await res.json()
        const items = Array.isArray(data?.stops) ? data.stops : []
        if (!cancelled) {
          setSearchResults({ loading: false, error: '', items })
        }
      } catch (error) {
        if (!cancelled) {
          setSearchResults({
            loading: false,
            error: error instanceof Error ? error.message : 'Could not search stops',
            items: [],
          })
        }
      }
    }

    loadSearchResults()
    setDropdownOpen(!!query.trim())
    return () => {
      cancelled = true
    }
  }, [query])

  const visibleStops = useMemo(() => {
    if (mapZoom < STATION_LOAD_ZOOM || stops.items.length === 0) {
      return []
    }

    const maxDistanceKm = Math.max(0.45, MAX_RENDER_DISTANCE_KM - (MAP_MAX_ZOOM - mapZoom) * 0.25)
    return stops.items.filter((stop) => {
      const distanceKm = haversineKm(mapCenter[0], mapCenter[1], stop.lat, stop.lon)
      return distanceKm <= maxDistanceKm
    })
  }, [mapCenter, mapZoom, stops.items])

  return (
    <main className="landing-shell">
      <section className="hero-band">
        <p className="eyebrow">ForeTransit</p>
        <h1>TTC Live Map</h1>
        <p className="hero-copy">
          Real-world geography. Stops only appear after you zoom in, and nearby ones are clickable.
        </p>
        <div className="hero-actions">
          <div className="search-stack" ref={searchRef}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search station names"
              onFocus={() => setDropdownOpen(!!query.trim())}
            />
            {dropdownOpen && query.trim() ? (
              <div className="search-results">
                {searchResults.loading ? <span className="search-hint">Searching...</span> : null}
                {searchResults.error ? <span className="mini-error">{searchResults.error}</span> : null}
                {!searchResults.loading && searchResults.items.length === 0 ? (
                  <span className="search-hint">No station matches found</span>
                ) : null}
                {searchResults.items.map((stop) => (
                  <button
                    key={stop.stop_id}
                    type="button"
                    className="search-result-item"
                    onClick={() => {
                      onSelectStop(stop)
                      setFocusStop(stop)
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
          <button type="button" onClick={onEnterForecast} disabled={!selectedStop}>
            {selectedStop ? `Open Forecast for ${selectedStop.stop_name}` : 'Pick a station'}
          </button>
        </div>
        {stops.error ? <p className="error">{stops.error}</p> : null}
      </section>

      <section className="map-panel">
        <MapContainer
          center={[43.6532, -79.3832]}
          zoom={11}
          minZoom={MAP_MIN_ZOOM}
          maxZoom={MAP_MAX_ZOOM}
          className="ttc-map"
          scrollWheelZoom
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          <ZoomTracker
            onViewChange={(zoom, center) => {
              setMapZoom(zoom)
              setMapCenter([center.lat, center.lng])
            }}
          />

          <FocusMap stop={focusStop} />

          {mapZoom >= STATION_LOAD_ZOOM
            ? visibleStops.map((stop) => {
                const active = selectedStop?.stop_id === stop.stop_id
                const opacity = Math.max(
                  0,
                  Math.min((mapZoom - STATION_FADE_START_ZOOM) / (MAP_MAX_ZOOM - STATION_FADE_START_ZOOM), 1),
                )
                return (
                  <Marker
                    key={stop.stop_id}
                    position={[stop.lat, stop.lon]}
                    icon={stationIcon(stop.mode, active, opacity)}
                    eventHandlers={{
                      click: () => onSelectStop(stop),
                    }}
                  >
                    <Popup>
                      <div className="popup-card">
                        <strong>{stop.stop_name}</strong>
                        <p>ID: {stop.stop_id}</p>
                        <button type="button" onClick={() => onSelectStop(stop)}>
                          Open station
                        </button>
                      </div>
                    </Popup>
                  </Marker>
                )
              })
            : null}
        </MapContainer>

        <div className="map-meta">
          <span>
            {stops.loading
              ? 'Loading TTC stops...'
              : mapZoom >= STATION_LOAD_ZOOM
                ? `${visibleStops.length} nearby stops visible`
                : 'Zoom in to load nearby stops'}
          </span>
          <span>{selectedStop ? `Selected: ${selectedStop.stop_name}` : 'No station selected'}</span>
        </div>
      </section>
    </main>
  )
}

function ForecastDashboard({ selectedStop, onBackToMap }) {
  const [stopSearch, setStopSearch] = useState(selectedStop?.stop_name || 'Danforth')
  const [stopId, setStopId] = useState(selectedStop?.stop_id || '')
  const [stops, setStops] = useState({ loading: false, error: '', items: [] })
  const [health, setHealth] = useState({ loading: true, ok: false })
  const [vehicles, setVehicles] = useState({ loading: true, count: 0, error: '' })
  const [forecast, setForecast] = useState({ loading: false, data: null, error: '' })

  useEffect(() => {
    if (selectedStop?.stop_id) {
      setStopId(selectedStop.stop_id)
      setStopSearch(selectedStop.stop_name || '')
    }
  }, [selectedStop])

  useEffect(() => {
    let cancelled = false

    async function checkHealth() {
      setHealth({ loading: true, ok: false, message: '' })
      try {
        const res = await fetch(`${API_ROOT}/`)
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const data = await res.json()
        if (!cancelled) {
          setHealth({ loading: false, ok: true, message: data.status || 'Backend online' })
        }
      } catch (error) {
        if (!cancelled) {
          setHealth({
            loading: false,
            ok: false,
            message: error instanceof Error ? error.message : 'Request failed',
          })
        }
      }
    }

    checkHealth()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function loadStops() {
      setStops((prev) => ({ ...prev, loading: true, error: '' }))
      try {
        const query = encodeURIComponent(stopSearch.trim())
        const res = await fetch(`${API_ROOT}/stops/search?q=${query}&limit=40`)
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const data = await res.json()
        const items = Array.isArray(data?.stops) ? data.stops : []
        if (!cancelled) {
          setStops({ loading: false, error: '', items })
          if (items.length > 0 && !items.some((s) => s.stop_id === stopId)) {
            setStopId(items[0].stop_id)
          }
        }
      } catch (error) {
        if (!cancelled) {
          setStops({
            loading: false,
            error: error instanceof Error ? error.message : 'Could not load stops',
            items: [],
          })
        }
      }
    }

    loadStops()
    return () => {
      cancelled = true
    }
  }, [stopSearch, stopId])

  useEffect(() => {
    let cancelled = false

    async function getVehicles() {
      setVehicles((prev) => ({ ...prev, loading: true, error: '' }))
      try {
        const res = await fetch(`${API_ROOT}/vehicles/live`)
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const data = await res.json()
        const count = Array.isArray(data) ? data.length : 0
        if (!cancelled) {
          setVehicles({ loading: false, count, error: '' })
        }
      } catch (error) {
        if (!cancelled) {
          setVehicles({
            loading: false,
            count: 0,
            error: error instanceof Error ? error.message : 'Could not load vehicles',
          })
        }
      }
    }

    getVehicles()
    const timer = setInterval(getVehicles, 30000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [])

  async function handleSubmit(event) {
    event.preventDefault()
    const cleanedStopId = stopId.trim()

    if (!cleanedStopId) {
      setForecast({ loading: false, data: null, error: 'Please select a stop name.' })
      return
    }

    setForecast({ loading: true, data: null, error: '' })

    try {
      const res = await fetch(`${API_ROOT}/station/${encodeURIComponent(cleanedStopId)}`)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }

      const data = await res.json()
      if (data?.error) {
        throw new Error(data.error)
      }

      setForecast({ loading: false, data, error: '' })
    } catch (error) {
      setForecast({
        loading: false,
        data: null,
        error: error instanceof Error ? error.message : 'Could not fetch forecast',
      })
    }
  }

  const current = forecast.data?.current
  const topFactors = Array.isArray(forecast.data?.top_factors) ? forecast.data.top_factors : []
  const nearbyVehicles = Array.isArray(forecast.data?.vehicles_nearby)
    ? forecast.data.vehicles_nearby
    : []
  const weather = forecast.data?.weather
  const warnings = Array.isArray(forecast.data?.warnings) ? forecast.data.warnings : []

  const selectedOptionMissing = useMemo(() => {
    if (!stopId) {
      return false
    }
    return !stops.items.some((s) => s.stop_id === stopId)
  }, [stops.items, stopId])

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">ForeTransit</p>
          <h1>Transit Delay Prediction</h1>
        </div>
        <div className="topbar-actions">
          <div className={`health ${statusClass(health)}`}>
            <span className="dot" aria-hidden="true" />
            <span>
              {health.loading ? 'Checking API...' : health.ok ? 'API Online' : 'API Offline'}
            </span>
          </div>
          <button type="button" className="ghost-btn" onClick={onBackToMap}>
            Back to Map
          </button>
        </div>
      </header>

      <section className="grid">
        <article className="panel form-panel">
          <h2>Query Forecast</h2>
          <p className="panel-copy">Search by station name and fetch prediction automatically.</p>

          <form className="form" onSubmit={handleSubmit}>
            <label htmlFor="stop-search">Stop name search</label>
            <input
              id="stop-search"
              type="text"
              value={stopSearch}
              onChange={(e) => setStopSearch(e.target.value)}
              placeholder="Danforth"
            />

            <label htmlFor="stop-select">Select stop</label>
            <select
              id="stop-select"
              value={stopId}
              onChange={(e) => setStopId(e.target.value)}
              disabled={stops.loading || stops.items.length === 0}
            >
              {selectedOptionMissing && selectedStop ? (
                <option value={selectedStop.stop_id}>{selectedStop.stop_name}</option>
              ) : null}
              {stops.items.length === 0 ? (
                <option value="">No stops found</option>
              ) : (
                stops.items.map((stop) => (
                  <option key={stop.stop_id} value={stop.stop_id}>
                    {stop.stop_name}
                  </option>
                ))
              )}
            </select>

            {stops.error ? <p className="mini-error">{stops.error}</p> : null}

            <button type="submit" disabled={forecast.loading}>
              {forecast.loading ? 'Loading...' : 'Get Forecast'}
            </button>
          </form>

          {forecast.error ? <p className="error">{forecast.error}</p> : null}
          {warnings.map((warning) => (
            <p key={warning} className="mini-warning">
              {warning}
            </p>
          ))}
        </article>

        <article className="panel stat-panel">
          <h2>Live Snapshot</h2>
          <div className="stats">
            <div className="stat-card">
              <span className="label">Vehicles live</span>
              <span className="value">{vehicles.loading ? '...' : vehicles.count}</span>
              {vehicles.error ? <span className="mini-error">{vehicles.error}</span> : null}
            </div>
            <div className="stat-card">
              <span className="label">Station</span>
              <span className="value small">{forecast.data?.stop_name || 'Select stop'}</span>
            </div>
            <div className="stat-card">
              <span className="label">Predicted delay</span>
              <span className="value small">{formatDelay(current?.delay_min)}</span>
            </div>
            <div className="stat-card">
              <span className="label">Risk label</span>
              <span className="value small">{current?.label || 'N/A'}</span>
            </div>
          </div>
        </article>
      </section>

      <section className="grid second">
        <article className="panel wide">
          <h2>Top Delay Factors</h2>
          {topFactors.length === 0 ? (
            <p className="panel-copy">Fetch a forecast to see contributing factors.</p>
          ) : (
            <ul className="factor-list">
              {topFactors.map((factor, idx) => (
                <li key={`${factor.factor}-${idx}`}>
                  <div>
                    <p>{factor.factor}</p>
                    <span className="impact">Impact: {factor.impact}</span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </article>

        <article className="panel wide">
          <h2>Nearby Vehicles</h2>
          {nearbyVehicles.length === 0 ? (
            <p className="panel-copy">No nearby vehicles shown yet.</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Vehicle</th>
                    <th>Route</th>
                    <th>Delay (sec)</th>
                    <th>Speed (km/h)</th>
                  </tr>
                </thead>
                <tbody>
                  {nearbyVehicles.slice(0, 8).map((vehicle) => (
                    <tr key={`${vehicle.vehicle_id}-${vehicle.route_id}`}>
                      <td>{vehicle.vehicle_id}</td>
                      <td>{vehicle.route_id}</td>
                      <td>{vehicle.delay_seconds}</td>
                      <td>{vehicle.speed_kmh}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </article>
      </section>

      <footer className="footer">
        <p>
          {weather
            ? `Weather: ${weather.temperature}C, rain ${weather.rain}mm, snow ${weather.snow}mm, wind ${weather.wind}km/h`
            : 'Weather data appears after station lookup.'}
        </p>
      </footer>
    </main>
  )
}

function App() {
  const [view, setView] = useState('map')
  const [selectedStop, setSelectedStop] = useState(null)

  if (view === 'forecast') {
    return (
      <ForecastDashboard
        selectedStop={selectedStop}
        onBackToMap={() => setView('map')}
      />
    )
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