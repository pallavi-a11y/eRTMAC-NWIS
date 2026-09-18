import { useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  ImageOverlay,
  Marker,
  Popup,
  Circle,
  ZoomControl,
  useMap,
  useMapEvents,
} from "react-leaflet";
import L from "leaflet";

import {
  Search,
  MapPin,
  Navigation,
  Crosshair,
  X,
  Activity,
  History,
  Radio,
  LocateFixed,
  CircleDot,
  Ruler,
  Layers,
  Plus,
  Gauge,
  ShieldCheck,
  ShieldQuestion,
} from "lucide-react";

import "leaflet/dist/leaflet.css";
import "./MapVisualise.css";

import api from "../api/client";
import indiaBasemap from "../assets/india-basemap.jpg";
import { INDIA_PLACES } from "../data/indiaPlaces";

/* =========================================================
   OFFLINE BASEMAP
   A static, locally-stored India map image is used instead of
   live internet map tiles (Esri/Google/etc.) so this page keeps
   working with no network connection at all - required since the
   whole product must run fully offline. The image is calibrated
   to the standard India location-map bounds (Wikipedia's "India
   location map" template: 5.0N-37.5N, 67.0E-99.0E), so plotting
   a well's real lat/lng on top of it lines up correctly.
========================================================= */

const INDIA_IMAGE_BOUNDS = [
  [5.0, 67.0],
  [37.5, 99.0],
];

const MAP_MIN_ZOOM = 4;
const MAP_MAX_ZOOM = 10;

const placeLabelIcon = (name) =>
  L.divIcon({
    className: "place-label-wrapper",
    html: `<span class="place-label">${name}</span>`,
    iconSize: [0, 0],
    iconAnchor: [0, 0],
  });

/* =========================================================
   PLACE LABELS
   Real city/state-capital names revealed progressively as the
   user zooms in, so the offline basemap reads like an actual
   map rather than a bare outline with only well pins on it.
========================================================= */

const PlaceLabels = () => {
  const map = useMap();
  const [zoom, setZoom] = useState(map.getZoom());

  useEffect(() => {
    const handleZoom = () => setZoom(map.getZoom());
    map.on("zoomend", handleZoom);
    return () => map.off("zoomend", handleZoom);
  }, [map]);

  return INDIA_PLACES.filter((place) => zoom >= place.minZoom).map(
    (place) => (
      <Marker
        key={place.name}
        position={[place.lat, place.lng]}
        icon={placeLabelIcon(place.name)}
        interactive={false}
      />
    )
  );
};

/* =========================================================
   DISTANCE CALCULATION (haversine, purely local - no API call)
========================================================= */

/* Several real wells only have a partial extraction from their source
   document (e.g. longitude read but latitude missed) - lat/lng can be
   null. Those can't be plotted or measured, so every place that touches
   coordinates checks this first instead of assuming both are numbers. */
const hasCoords = (well) =>
  typeof well.lat === "number" && typeof well.lng === "number";

const calculateDistance = (lat1, lon1, lat2, lon2) => {
  const R = 6371;

  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;

  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c;
};

/* =========================================================
   CUSTOM WELL ICON
   Real lifecycle status (ACTIVE vs PLUGGED/SUSPENDED/UNKNOWN)
   drives the marker styling - reusing the existing
   active/historical CSS classes rather than renaming them.
========================================================= */

const createWellIcon = (isActive) => {
  return L.divIcon({
    className: "custom-well-marker-wrapper",

    html: `
      <div class="custom-well-marker ${
        isActive ? "active-marker" : "historical-marker"
      }">
        ${
          isActive
            ? '<span class="marker-pulse"></span>'
            : ""
        }

        <div class="marker-inner">
          <span class="marker-dot"></span>
        </div>
      </div>
    `,

    iconSize: [34, 34],
    iconAnchor: [17, 17],
    popupAnchor: [0, -17],
  });
};

/* =========================================================
   MAP FOCUS
========================================================= */

const MapFocus = ({ well }) => {
  const map = useMap();

  useEffect(() => {
    if (well && hasCoords(well)) {
      map.flyTo([well.lat, well.lng], 7, {
        duration: 1.2,
      });
    }
  }, [well, map]);

  return null;
};

/* =========================================================
   MAP CLICK HANDLER
========================================================= */

const MapClickHandler = ({ showRadius, onMapClick }) => {
  useMapEvents({
    click(e) {
      if (showRadius) {
        onMapClick([e.latlng.lat, e.latlng.lng]);
      }
    },
  });

  return null;
};

/* =========================================================
   MAP RESIZER
========================================================= */

const MapResizer = () => {
  const map = useMap();

  useEffect(() => {
    const timer = setTimeout(() => {
      map.invalidateSize();
    }, 200);

    return () => clearTimeout(timer);
  }, [map]);

  return null;
};

/* =========================================================
   MAIN COMPONENT
========================================================= */

const MapVisualise = () => {
  const [wells, setWells] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(null);

  const [search, setSearch] = useState("");

  const [radiusInput, setRadiusInput] = useState("");

  const [radius, setRadius] = useState(null);

  const [radiusCenter, setRadiusCenter] = useState(null);

  const [selectedWell, setSelectedWell] = useState(null);

  const [showRadius, setShowRadius] = useState(false);

  /* Add-well form */
  const [showAddWell, setShowAddWell] = useState(false);

  const [newWell, setNewWell] = useState({
    status: "ACTIVE",
    wellType: "",
    name: "",
    latitude: "",
    longitude: "",
  });

  const [savingWell, setSavingWell] = useState(false);

  /* =====================================================
     LOAD REAL WELLS
  ===================================================== */

  useEffect(() => {
    let cancelled = false;

    api
      .get("/api/wells")
      .then((res) => {
        if (!cancelled) {
          setWells(res.data);
          setLoadError(null);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setLoadError(
            "Could not reach the backend. Is the server running?"
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  /* =====================================================
     SEARCH + RADIUS FILTER
     Both the sidebar list and the map markers share this
     same filtered set, so "locate the wells in the radius"
     means the same thing everywhere on the page.
  ===================================================== */

  const filteredWells = useMemo(() => {
    const value = search.toLowerCase().trim();

    let result = wells;

    if (value) {
      result = result.filter(
        (well) =>
          well.name.toLowerCase().includes(value) ||
          String(well.id).toLowerCase().includes(value) ||
          well.location.toLowerCase().includes(value) ||
          well.type.toLowerCase().includes(value)
      );
    }

    if (showRadius && radius && radiusCenter) {
      result = result.filter(
        (well) =>
          hasCoords(well) &&
          calculateDistance(
            radiusCenter[0],
            radiusCenter[1],
            well.lat,
            well.lng
          ) <= radius
      );
    }

    return result;
  }, [search, wells, showRadius, radius, radiusCenter]);

  /* =====================================================
     RADIUS
  ===================================================== */

  const applyRadius = () => {
    const value = Number(radiusInput);

    if (!value || value <= 0) {
      alert("Please enter a valid radius greater than 0 km.");
      return;
    }

    setRadius(value);
    setShowRadius(true);

    // Centers on whichever well you've actually selected (list click, marker
    // click, or the Active Well shortcut) - falls back to the map's last
    // clicked point if one exists, and only reaches for an arbitrary active
    // well as a last resort when nothing has been picked yet at all. Found
    // as a real bug: this used to always re-center on wells.find(ACTIVE) -
    // the SAME well every time regardless of what you'd selected, so "show
    // me wells around this one" never actually worked for any well other
    // than whichever happened to be first in the array.
    if (!radiusCenter) {
      if (selectedWell && hasCoords(selectedWell)) {
        setRadiusCenter([selectedWell.lat, selectedWell.lng]);
      } else {
        const activeWell = wells.find(
          (well) => well.status === "ACTIVE" && hasCoords(well)
        );

        if (activeWell) {
          setRadiusCenter([
            activeWell.lat,
            activeWell.lng,
          ]);
        }
      }
    }
  };

  const clearRadius = () => {
    setRadius(null);
    setRadiusInput("");
    setRadiusCenter(null);
    setShowRadius(false);
  };

  /* =====================================================
     ACTIVE WELL
  ===================================================== */

  const useActiveWell = () => {
    const activeWell = wells.find(
      (well) => well.status === "ACTIVE" && hasCoords(well)
    );

    if (activeWell) {
      setSelectedWell(activeWell);

      setRadiusCenter([
        activeWell.lat,
        activeWell.lng,
      ]);
    }
  };

  /* =====================================================
     SELECT ANY WELL
     The single entry point for "pick a well" from either the sidebar list
     or a map marker. If a radius search is already running, moves it to
     center on whatever well was just picked - so "select any well and see
     the wells around it" is one click, not select-well-then-separately-
     re-apply-radius.
  ===================================================== */

  const selectWell = (well) => {
    setSelectedWell(well);

    if (showRadius && hasCoords(well)) {
      setRadiusCenter([well.lat, well.lng]);
    }
  };

  /* =====================================================
     MAP CLICK
  ===================================================== */

  const handleMapClick = (coordinates) => {
    if (showRadius) {
      setRadiusCenter(coordinates);
    }
  };

  /* =====================================================
     ADD WELL - persists for real via POST /api/wells
  ===================================================== */

  const addWell = async (e) => {
    e.preventDefault();

    const name = newWell.name.trim();
    const lat = Number(newWell.latitude);
    const lng = Number(newWell.longitude);

    if (!name) {
      alert("Please enter the well name.");
      return;
    }

    if (
      newWell.latitude === "" ||
      Number.isNaN(lat) ||
      lat < -90 ||
      lat > 90
    ) {
      alert("Please enter a valid latitude between -90 and 90.");
      return;
    }

    if (
      newWell.longitude === "" ||
      Number.isNaN(lng) ||
      lng < -180 ||
      lng > 180
    ) {
      alert("Please enter a valid longitude between -180 and 180.");
      return;
    }

    setSavingWell(true);

    try {
      const res = await api.post("/api/wells", {
        name,
        latitude: lat,
        longitude: lng,
        well_type: newWell.wellType.trim() || null,
        status: newWell.status,
      });

      const saved = res.data;

      setWells((prev) => [...prev, saved]);
      setSelectedWell(saved);
      setRadiusCenter([lat, lng]);
      setShowAddWell(false);

      setNewWell({
        status: "ACTIVE",
        wellType: "",
        name: "",
        latitude: "",
        longitude: "",
      });
    } catch {
      alert("Could not save this well. Is the backend running?");
    } finally {
      setSavingWell(false);
    }
  };

  /* =====================================================
     COUNTS
  ===================================================== */

  const activeCount = wells.filter(
    (well) => well.status === "ACTIVE"
  ).length;

  const inactiveCount = wells.length - activeCount;

  /* =====================================================
     RETURN
  ===================================================== */

  return (
    <div className="map-page">

      {/* =================================================
          LEFT CONTROL PANEL
      ================================================= */}

      <aside className="map-control-panel">

        {/* BRAND */}

        <div className="map-brand">
          <div className="map-brand-icon">
            <Gauge size={23} />
          </div>

          <div>
            <h2>eRTMAC-NWIS</h2>
            <span>Well Map Visualiser</span>
          </div>
        </div>

        {/* STATS */}

        <div className="well-stats">

          <div className="well-stat active-stat">
            <Activity size={18} />

            <div>
              <strong>{activeCount}</strong>
              <span>Active Wells</span>
            </div>
          </div>

          <div className="well-stat historical-stat">
            <History size={18} />

            <div>
              <strong>{inactiveCount}</strong>
              <span>Inactive</span>
            </div>
          </div>

        </div>

        {/* SEARCH */}

        <div className="map-search">
          <Search size={18} />

          <input
            type="text"
            placeholder="Search wells..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />

          {search && (
            <button
              type="button"
              className="clear-search"
              onClick={() => setSearch("")}
            >
              <X size={15} />
            </button>
          )}
        </div>

        {/* RADIUS */}

        <div className="radius-panel">

          <div className="section-title">

            <div>
              <CircleDot size={17} />
              <span>Radius Search</span>
            </div>

            {showRadius && (
              <button
                type="button"
                className="small-clear-btn"
                onClick={clearRadius}
              >
                Clear
              </button>
            )}

          </div>

          <div className="radius-input-row">

            <input
              type="number"
              min="1"
              placeholder="Radius in km"
              value={radiusInput}
              onChange={(e) =>
                setRadiusInput(e.target.value)
              }
            />

            <button
              type="button"
              className="radius-apply-btn"
              onClick={applyRadius}
            >
              Apply
            </button>

          </div>

          <div className="radius-help">
            <Crosshair size={14} />

            <span>
              Click anywhere on the map to move the radius center.
              {showRadius && radius
                ? ` Showing ${filteredWells.length} of ${wells.length} wells within ${radius} km.`
                : ""}
            </span>
          </div>

        </div>

        {/* =================================================
            ADD WELL
        ================================================= */}

        <div className="add-well-section">

          <button
            type="button"
            className="add-well-toggle"
            onClick={() =>
              setShowAddWell((prev) => !prev)
            }
          >
            <span>
              <Plus size={17} />
              Add Well
            </span>

            {showAddWell ? (
              <X size={16} />
            ) : (
              <Navigation size={16} />
            )}
          </button>

          {showAddWell && (
            <form
              className="add-well-form"
              onSubmit={addWell}
            >

              <div className="form-title">
                <MapPin size={16} />
                <span>Add Well Location</span>
              </div>

              <div className="form-field">
                <label>Well Name</label>

                <input
                  type="text"
                  placeholder="Enter well name"
                  value={newWell.name}
                  onChange={(e) =>
                    setNewWell({
                      ...newWell,
                      name: e.target.value,
                    })
                  }
                />
              </div>

              <div className="form-field">
                <label>Status</label>

                <select
                  value={newWell.status}
                  onChange={(e) =>
                    setNewWell({
                      ...newWell,
                      status: e.target.value,
                    })
                  }
                >
                  <option value="ACTIVE">Active</option>
                  <option value="SUSPENDED">Suspended</option>
                  <option value="PLUGGED">Plugged</option>
                  <option value="UNKNOWN">Unknown</option>
                </select>
              </div>

              <div className="form-field">
                <label>Well Type (optional)</label>

                <input
                  type="text"
                  placeholder="e.g. Oil Producer"
                  value={newWell.wellType}
                  onChange={(e) =>
                    setNewWell({
                      ...newWell,
                      wellType: e.target.value,
                    })
                  }
                />
              </div>

              <div className="coordinate-inputs">

                <div className="form-field">
                  <label>Latitude</label>

                  <input
                    type="number"
                    step="any"
                    placeholder="30.3165"
                    value={newWell.latitude}
                    onChange={(e) =>
                      setNewWell({
                        ...newWell,
                        latitude: e.target.value,
                      })
                    }
                  />
                </div>

                <div className="form-field">
                  <label>Longitude</label>

                  <input
                    type="number"
                    step="any"
                    placeholder="78.0322"
                    value={newWell.longitude}
                    onChange={(e) =>
                      setNewWell({
                        ...newWell,
                        longitude: e.target.value,
                      })
                    }
                  />
                </div>

              </div>

              <button
                type="submit"
                className="save-well-btn"
                disabled={savingWell}
              >
                <MapPin size={16} />
                {savingWell ? "Saving..." : "Save Well"}
              </button>

            </form>
          )}

        </div>

        {/* WELL LIST HEADER */}

        <div className="well-list-header">

          <div>
            <span>Well Network</span>

            <small>
              {filteredWells.length} of {wells.length}
            </small>
          </div>

          <Radio size={17} />

        </div>

        {/* WELL LIST */}

        <div className="well-list">

          {loading ? (

            <div className="no-wells">
              <Radio size={25} />
              <p>Loading wells...</p>
            </div>

          ) : loadError ? (

            <div className="no-wells">
              <ShieldQuestion size={25} />
              <p>{loadError}</p>
              <span>Start the backend, then refresh this page.</span>
            </div>

          ) : filteredWells.length === 0 ? (

            <div className="no-wells">
              <Search size={25} />
              <p>No wells found</p>
              <span>
                {wells.length === 0
                  ? "No wells in the database yet."
                  : "Try another search or radius."}
              </span>
            </div>

          ) : (

            filteredWells.map((well) => (

              <button
                type="button"
                key={well.id}
                className={`well-card ${
                  selectedWell?.id === well.id
                    ? "selected"
                    : ""
                }`}
                onClick={() =>
                  selectWell(well)
                }
              >

                <div
                  className={`well-type-dot ${
                    well.status === "ACTIVE"
                      ? "active-dot"
                      : "historical-dot"
                  }`}
                />

                <div className="well-card-content">

                  <div className="well-card-top">

                    <strong>{well.name}</strong>

                    <span
                      className={`well-badge ${
                        well.status === "ACTIVE"
                          ? "active-badge"
                          : "historical-badge"
                      }`}
                    >
                      {well.status}
                    </span>

                  </div>

                  <div className="well-card-id">
                    {well.type}
                  </div>

                  <div className="well-card-location">
                    <MapPin size={12} />
                    {well.location}
                    {!hasCoords(well) && " · no coordinates"}
                  </div>

                </div>

              </button>

            ))

          )}

        </div>

        {/* FOOTER */}

        <div className="map-panel-footer">
          <span>
            <LocateFixed size={14} />
            Offline GIS Map
          </span>
        </div>

      </aside>

      {/* =================================================
          MAP AREA
      ================================================= */}

      <main className="map-area">

        <MapContainer
          center={[22.0, 82.5]}
          zoom={5}
          minZoom={MAP_MIN_ZOOM}
          maxZoom={MAP_MAX_ZOOM}
          maxBounds={INDIA_IMAGE_BOUNDS}
          maxBoundsViscosity={0.8}
          zoomControl={false}
          className="leaflet-map"
        >

          <MapResizer />

          <MapFocus well={selectedWell} />

          <MapClickHandler
            showRadius={showRadius}
            onMapClick={handleMapClick}
          />

          {/* OFFLINE BASEMAP - a locally-stored image, no network required */}

          <ImageOverlay
            url={indiaBasemap}
            bounds={INDIA_IMAGE_BOUNDS}
          />

          <PlaceLabels />

          <ZoomControl position="bottomright" />

          {/* RADIUS */}

          {showRadius &&
            radius &&
            radiusCenter && (
              <Circle
                center={radiusCenter}
                radius={radius * 1000}
                pathOptions={{
                  color: "#2563eb",
                  fillColor: "#2563eb",
                  fillOpacity: 0.12,
                  weight: 2,
                }}
              />
            )}

          {/* WELL MARKERS - dimmed when outside an active radius search */}

          {wells.filter(hasCoords).map((well) => {

            const inRadius = filteredWells.some(
              (item) => item.id === well.id
            );

            const dimmed =
              showRadius && radius && radiusCenter && !inRadius;

            return (
              <Marker
                key={well.id}
                position={[well.lat, well.lng]}
                icon={createWellIcon(well.status === "ACTIVE")}
                opacity={dimmed ? 0.28 : 1}
                eventHandlers={{
                  click: () =>
                    selectWell(well),
                }}
              >

                <Popup
                  offset={[0, -14]}
                >

                  <div className="well-popup">

                    <div className="popup-header">

                      <span
                        className={`popup-type ${
                          well.status === "ACTIVE"
                            ? "popup-active"
                            : "popup-history"
                        }`}
                      >
                        {well.status}
                      </span>

                      <h3>{well.name}</h3>

                      <small>{well.type}</small>

                    </div>

                    <div className="popup-location">
                      <MapPin size={14} />
                      {well.location}
                    </div>

                    <div className="popup-coordinates">

                      <div>
                        <span>Latitude</span>

                        <strong>
                          {well.lat.toFixed(4)}
                        </strong>
                      </div>

                      <div>
                        <span>Longitude</span>

                        <strong>
                          {well.lng.toFixed(4)}
                        </strong>
                      </div>

                    </div>

                    <div className="popup-details">

                      <div>
                        <Ruler size={14} />
                        <span>Depth</span>

                        <strong>
                          {well.depth
                            ? `${well.depth} m`
                            : "Unknown"}
                        </strong>
                      </div>

                      <div>
                        <Gauge size={14} />
                        <span>Operator</span>

                        <strong>
                          {well.operator || "Unknown"}
                        </strong>
                      </div>

                    </div>

                    <div className="popup-status">
                      <span>Coordinates</span>
                      <strong>
                        {well.location_verified ? (
                          <>
                            <ShieldCheck
                              size={11}
                              style={{ verticalAlign: "-2px", marginRight: 3 }}
                            />
                            Verified
                          </>
                        ) : (
                          <>
                            <ShieldQuestion
                              size={11}
                              style={{ verticalAlign: "-2px", marginRight: 3 }}
                            />
                            Unverified
                          </>
                        )}
                      </strong>
                    </div>

                  </div>

                </Popup>

              </Marker>
            );
          })}

        </MapContainer>

        {/* =================================================
            MAP HEADER
        ================================================= */}

        <div className="map-top-overlay">

          <div>
            <h1>India Well Network</h1>
            <p>Offline spatial monitoring view</p>
          </div>

          <div className="map-top-actions">

            {/* ACTIVE WELL */}

            <button
              type="button"
              className="active-well-btn"
              onClick={useActiveWell}
            >
              <LocateFixed size={16} />
              Active Well
            </button>

          </div>

        </div>

        {/* =================================================
            LEGEND
        ================================================= */}

        <div className="map-legend">

          <div className="legend-title">
            <Layers size={15} />
            <span>Legend</span>
          </div>

          <div className="legend-item">
            <span className="legend-marker active-legend" />
            <span>Active Well</span>
          </div>

          <div className="legend-item">
            <span className="legend-marker historical-legend" />
            <span>Inactive (Plugged / Suspended)</span>
          </div>

          {showRadius &&
            radius &&
            radiusCenter && (
              <div className="legend-item">
                <span className="radius-legend" />
                <span>Radius: {radius} km</span>
              </div>
            )}

        </div>

        {/* =================================================
            COORDINATE CARD
        ================================================= */}

        {radiusCenter && (

          <div className="coordinate-card">

            <div className="coordinate-card-icon">
              <Crosshair size={17} />
            </div>

            <div>
              <span>Map Center</span>

              <strong>
                {radiusCenter[0].toFixed(5)},{" "}
                {radiusCenter[1].toFixed(5)}
              </strong>
            </div>

            <button
              type="button"
              onClick={() =>
                setRadiusCenter(null)
              }
              title="Clear center"
            >
              <X size={15} />
            </button>

          </div>

        )}

      </main>
    </div>
  );
};

export default MapVisualise;
