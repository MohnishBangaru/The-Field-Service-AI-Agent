from __future__ import annotations

import datetime as _dt
from typing import Optional, List, Dict, Any
import re

import httpx
from langchain_core.tools import tool
import math
from .errors import ToolError
from bs4 import BeautifulSoup
try:
	from ddgs import DDGS  # new package name
except Exception:  # fallback for older envs
	from duckduckgo_search import DDGS
from .settings import load_settings


@tool
def get_time(tz: Optional[str] = None) -> str:
	"""Return the current time. Optionally pass IANA timezone like 'UTC' or 'America/Los_Angeles'."""
	if tz:
		try:
			import zoneinfo  # py3.9+
			tzinfo = zoneinfo.ZoneInfo(tz)
		except Exception:
			tzinfo = _dt.timezone.utc
	else:
		tzinfo = _dt.timezone.utc
	return _dt.datetime.now(tzinfo).isoformat()


@tool
def http_get(url: str, timeout_s: float = 10.0) -> str:
	"""Fetch text content from a URL using HTTP GET."""
	try:
		resp = httpx.get(url, timeout=timeout_s)
		resp.raise_for_status()
		return resp.text[:5000]
	except Exception as e:
		raise ToolError(str(e))


@tool
def web_search(query: str, max_results: int = 5) -> str:
	"""Search the web using DuckDuckGo and return top result titles and links."""
	try:
		results = []
		with DDGS() as ddgs:
			for r in ddgs.text(query, max_results=max_results):
				title = r.get("title", "")
				link = r.get("href") or r.get("link") or ""
				if link:
					results.append(f"{title} — {link}")
		return "\n".join(results) if results else "No results found."
	except Exception as e:
		raise ToolError(str(e))


@tool
def web_fetch(url: str, timeout_s: float = 15.0, max_chars: int = 5000) -> str:
	"""Fetch a web page and return visible text content (truncated)."""
	try:
		resp = httpx.get(url, timeout=timeout_s, headers={"User-Agent": "hiya-voice-agent/0.1"})
		resp.raise_for_status()
		soup = BeautifulSoup(resp.text, "html.parser")
		for tag in soup(["script", "style", "noscript"]):
			tag.decompose()
		text = " ".join(soup.get_text(separator=" ").split())
		return text[:max_chars]
	except Exception as e:
		raise ToolError(str(e))


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	# Distance in meters
	R = 6371000.0
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlambda = math.radians(lon2 - lon1)
	a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	return R * c


def _compute_departure_time_rfc3339(lat: float, lng: float, minutes_ahead: int, api_key: str) -> str:
	"""Compute a future RFC3339 UTC timestamp based on local time at given lat/lng using Google Time Zone API."""
	now_utc = _dt.datetime.now(_dt.timezone.utc)
	try:
		resp = httpx.get(
			"https://maps.googleapis.com/maps/api/timezone/json",
			params={"location": f"{lat},{lng}", "timestamp": int(now_utc.timestamp()), "key": api_key},
			timeout=10.0,
		)
		resp.raise_for_status()
		j = resp.json()
		raw = int(j.get("rawOffset", 0))
		dst = int(j.get("dstOffset", 0))
		offset = raw + dst
		local_now = now_utc + _dt.timedelta(seconds=offset)
		local_future = local_now + _dt.timedelta(minutes=minutes_ahead)
		future_utc = local_future - _dt.timedelta(seconds=offset)
	except Exception:
		# Fallback: simply use UTC + minutes_ahead
		future_utc = now_utc + _dt.timedelta(minutes=minutes_ahead)

	return future_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")


@tool
def search_nearby_places(query: str, near: str, radius_m: float = 1000.0, limit: int = 10) -> str:
	"""Search nearby places for a query near a location string.

	Args:
	- query: free text like 'coffee', 'pharmacy', 'park'.
	- near: location string, e.g., 'Mission District, San Francisco' or '94016'.
	- radius_m: search radius in meters (default 1000).
	- limit: max results to return (default 10).

	Returns: A concise newline-separated list of 'Name — distance_m — url'.
	"""
	headers = {"User-Agent": "hiya-voice-agent/0.1 (+https://example.com)"}
	# Geocode with Nominatim
	geo_resp = httpx.get(
		"https://nominatim.openstreetmap.org/search",
		params={"q": near, "format": "json", "limit": 1},
		headers=headers,
		timeout=15.0,
	)
	geo_resp.raise_for_status()
	geo_data: List[Dict[str, Any]] = geo_resp.json()  # type: ignore[assignment]
	if not geo_data:
		return "No location found for 'near'."
	center_lat = float(geo_data[0]["lat"])  # type: ignore[index]
	center_lon = float(geo_data[0]["lon"])  # type: ignore[index]

	# Overpass query: search name/amenity/shop matches around point
	overpass_q = f"""
	[out:json][timeout:25];
	(
	  node(around:{int(radius_m)},{center_lat},{center_lon})["name"~"{query}",i];
	  way(around:{int(radius_m)},{center_lat},{center_lon})["name"~"{query}",i];
	  rel(around:{int(radius_m)},{center_lat},{center_lon})["name"~"{query}",i];
	  node(around:{int(radius_m)},{center_lat},{center_lon})["amenity"~"{query}",i];
	  node(around:{int(radius_m)},{center_lat},{center_lon})["shop"~"{query}",i];
	);
	out center;
	"""
	over_resp = httpx.post(
		"https://overpass-api.de/api/interpreter",
		data={"data": overpass_q},
		headers=headers,
		timeout=30.0,
	)
	over_resp.raise_for_status()
	ov = over_resp.json()
	elems: List[Dict[str, Any]] = ov.get("elements", [])
	if not elems:
		return "No nearby places found."

	results: List[Dict[str, Any]] = []
	for el in elems:
		tags = el.get("tags", {})
		name = tags.get("name") or tags.get("amenity") or tags.get("shop") or "(unnamed)"
		lat = el.get("lat")
		lon = el.get("lon")
		if lat is None or lon is None:
			center = el.get("center") or {}
			lat = center.get("lat")
			lon = center.get("lon")
		if lat is None or lon is None:
			continue
		dist_m = int(_haversine_m(center_lat, center_lon, float(lat), float(lon)))
		osm_type = el.get("type", "n")
		osm_id = el.get("id")
		if osm_type == "node":
			url = f"https://www.openstreetmap.org/node/{osm_id}"
		elif osm_type == "way":
			url = f"https://www.openstreetmap.org/way/{osm_id}"
		else:
			url = f"https://www.openstreetmap.org/relation/{osm_id}"
		results.append({"name": str(name), "dist": dist_m, "url": url})

	results.sort(key=lambda r: r["dist"])  # nearest first
	results = results[: max(1, int(limit))]
	lines = [f"{r['name']} — {r['dist']}m — {r['url']}" for r in results]
	return "\n".join(lines)


@tool
def google_places_search(query: str, location: str, radius_m: int = 1000, limit: int = 10) -> str:
	"""Search nearby places using Google Places Text Search around a location string.

	Args:
	- query: e.g., 'coffee', 'pharmacy'
	- location: free-text location (geocoded via Google Geocoding)
	- radius_m: search radius in meters
	- limit: max results

	Returns newline list 'Name — rating — address — url'. Requires GOOGLE_MAPS_API_KEY.
	"""
	settings = load_settings()
	api_key = settings.google_maps_api_key
	if not api_key:
		raise ToolError("Missing GOOGLE_MAPS_API_KEY")

	try:
		# Geocode location
		geo = httpx.get(
			"https://maps.googleapis.com/maps/api/geocode/json",
			params={"address": location, "key": api_key},
			timeout=15.0,
		)
		geo.raise_for_status()
		gj = geo.json()
		results = gj.get("results", [])
		if not results:
			return "No geocode result."
		loc = results[0]["geometry"]["location"]
		lat, lng = loc["lat"], loc["lng"]

		# Text search nearby
		search = httpx.get(
			"https://maps.googleapis.com/maps/api/place/textsearch/json",
			params={
				"query": query,
				"location": f"{lat},{lng}",
				"radius": radius_m,
				"key": api_key,
			},
			timeout=20.0,
		)
		search.raise_for_status()
		sj = search.json()
		items = sj.get("results", [])[: max(1, int(limit))]
		lines = []
		for it in items:
			name = it.get("name", "(unnamed)")
			rating = it.get("rating", "?")
			addr = it.get("formatted_address", it.get("vicinity", ""))
			place_id = it.get("place_id", "")
			url = f"https://www.google.com/maps/place/?q=place_id:{place_id}" if place_id else ""
			lines.append(f"{name} — {rating} — {addr} — {url}")
		return "\n".join(lines)
	except Exception as e:
		raise ToolError(str(e))


@tool
def google_optimize_route(addresses: List[str], mode: str = "driving", origin: Optional[str] = None, destination: Optional[str] = None) -> str:
	"""Optimize visit order for multiple addresses using Google Directions API (simple heuristic).

	Provide at least two addresses. Returns an ordered list with total distance/time.
	Requires GOOGLE_MAPS_API_KEY.
	"""
	settings = load_settings()
	api_key = settings.google_maps_api_key
	if not api_key:
		raise ToolError("Missing GOOGLE_MAPS_API_KEY")
	# Coerce various input forms into a clean list
	if isinstance(addresses, str):
		cand = [p.strip() for p in re.split(r"[\n;]|\s{2,}|,\s*(?=[A-Za-z])", addresses) if p.strip()]
		addresses = cand
	addresses = [a for a in (addresses or []) if isinstance(a, str) and a.strip()]
	if len(addresses) < 2:
		return "Please provide at least two addresses for route optimization."

	try:
		# Geocode origin/destination if provided, then addresses
		coords: List[Dict[str, Any]] = []
		# Helper to geocode one string
		def _geocode_one(addr: str) -> Optional[Dict[str, Any]]:
			g = httpx.get(
				"https://maps.googleapis.com/maps/api/geocode/json",
				params={"address": addr, "key": api_key},
				timeout=15.0,
			)
			g.raise_for_status()
			gj = g.json()
			res = gj.get("results", [])
			if not res:
				return None
			loc = res[0]["geometry"]["location"]
			return {"addr": addr, "lat": loc["lat"], "lng": loc["lng"]}

		start = _geocode_one(origin) if origin else None
		end = _geocode_one(destination) if destination else None

		for addr in addresses:
			geo = httpx.get(
				"https://maps.googleapis.com/maps/api/geocode/json",
				params={"address": addr, "key": api_key},
				timeout=15.0,
			)
			geo.raise_for_status()
			gj = geo.json()
			res = gj.get("results", [])
			if not res:
				# Skip addresses we can't geocode; continue collecting others
				continue
			loc = res[0]["geometry"]["location"]
			coords.append({"addr": addr, "lat": loc["lat"], "lng": loc["lng"]})

		# Ensure we have at least two geocoded points
		if len(coords) + (1 if start else 0) + (1 if end else 0) < 2:
			return "Could not geocode at least two addresses. Please refine the locations."

		# Build endpoints and intermediate waypoints
		_origin = start or (coords[0] if coords else None)
		_destination = end or (coords[-1] if coords else None)
		intermediates = [c for c in coords if c is not _origin and c is not _destination]
		if not _origin or not _destination:
			if not _origin:
				_origin = intermediates.pop(0)
			if not _destination:
				_destination = intermediates.pop(-1)

		# Step 1: get optimized waypoint order (no traffic preference)
		ordered = None
		if mode.lower() == "driving" and intermediates:
			body1 = {
				"origin": {"location": {"latLng": {"latitude": _origin["lat"], "longitude": _origin["lng"]}}},
				"destination": {"location": {"latLng": {"latitude": _destination["lat"], "longitude": _destination["lng"]}}},
				"travelMode": "DRIVE",
				"optimizeWaypointOrder": True,
			}
			if intermediates:
				body1["intermediates"] = [
					{"location": {"latLng": {"latitude": w["lat"], "longitude": w["lng"]}}}
					for w in intermediates
				]
			headers1 = {
				"X-Goog-Api-Key": api_key,
				"X-Goog-FieldMask": "routes.optimizedIntermediateWaypointIndex",
			}
			resp1 = httpx.post(
				"https://routes.googleapis.com/directions/v2:computeRoutes",
				json=body1,
				headers=headers1,
				timeout=20.0,
			)
			try:
				resp1.raise_for_status()
			except httpx.HTTPStatusError as he:
				msg = he.response.text
				raise ToolError(f"Routes API error {he.response.status_code}: {msg}")
			jr1 = resp1.json()
			routes1 = jr1.get("routes", [])
			opt_idx = (routes1[0].get("optimizedIntermediateWaypointIndex") if routes1 else None) or []
			if isinstance(opt_idx, list) and all(isinstance(i, int) for i in opt_idx):
				ordered = [_origin] + [intermediates[i] for i in opt_idx] + [_destination]

		# Fallback if no optimization performed
		if ordered is None:
			ordered = [_origin] + intermediates + [_destination]

		# Step 2: compute traffic-aware route for fixed order
		body2 = {
			"origin": {"location": {"latLng": {"latitude": ordered[0]["lat"], "longitude": ordered[0]["lng"]}}},
			"destination": {"location": {"latLng": {"latitude": ordered[-1]["lat"], "longitude": ordered[-1]["lng"]}}},
			"travelMode": "DRIVE" if mode.lower() == "driving" else mode.upper(),
		}
		if len(ordered) > 2:
			body2["intermediates"] = [
				{"location": {"latLng": {"latitude": w["lat"], "longitude": w["lng"]}}}
				for w in ordered[1:-1]
			]
		if mode.lower() == "driving":
			body2["routingPreference"] = "TRAFFIC_AWARE_OPTIMAL"
			body2["departureTime"] = _compute_departure_time_rfc3339(ordered[0]["lat"], ordered[0]["lng"], minutes_ahead=5, api_key=api_key)

		headers2 = {
			"X-Goog-Api-Key": api_key,
			"X-Goog-FieldMask": "routes.distanceMeters,routes.duration,routes.legs",
		}
		resp2 = httpx.post(
			"https://routes.googleapis.com/directions/v2:computeRoutes",
			json=body2,
			headers=headers2,
			timeout=20.0,
		)
		try:
			resp2.raise_for_status()
		except httpx.HTTPStatusError as he:
			msg = he.response.text
			raise ToolError(f"Routes API error {he.response.status_code}: {msg}")
		jr2 = resp2.json()
		routes2 = jr2.get("routes", [])
		if not routes2:
			return "No route found for the given locations."
		route2 = routes2[0]
		dist_m = route2.get("distanceMeters") or sum(leg.get("distanceMeters", 0) for leg in route2.get("legs", []))
		# Parse duration like "123s"
		dur_s = None
		dv = route2.get("duration")
		if isinstance(dv, str) and dv.endswith("s"):
			try:
				dur_s = int(float(dv[:-1]))
			except Exception:
				pass
		if dur_s is None:
			total = 0
			for leg in route2.get("legs", []):
				lv = leg.get("duration")
				if isinstance(lv, str) and lv.endswith("s"):
					try:
						total += int(float(lv[:-1]))
					except Exception:
						pass
			dur_s = total or None

		ordered_addrs = [c["addr"] for c in ordered if c]
		dist_km = int((dist_m or 0) / 1000)
		dur_min = int((dur_s or 0) / 60)
		return f" -> ".join(ordered_addrs) + f"\nTotal: {dist_km} km, {dur_min} min"
	except Exception as e:
		raise ToolError(str(e))


@tool
def google_directions(origin: str, destination: str, mode: str = "driving") -> str:
	"""Get step-by-step directions between two addresses using Google Directions API."""
	settings = load_settings()
	api_key = settings.google_maps_api_key
	if not api_key:
		raise ToolError("Missing GOOGLE_MAPS_API_KEY")
	params = {"origin": origin, "destination": destination, "mode": mode, "key": api_key}
	try:
		resp = httpx.get("https://maps.googleapis.com/maps/api/directions/json", params=params, timeout=20.0)
		resp.raise_for_status()
		j = resp.json()
		routes = j.get("routes", [])
		if not routes:
			return "No route found."
		legs = routes[0].get("legs", [])
		steps = []
		for leg in legs:
			for s in leg.get("steps", []):
				# strip HTML tags from instructions
				inst = re.sub(r"<[^>]+>", "", s.get("html_instructions", "")).strip()
				dist = s.get("distance", {}).get("text", "")
				dur = s.get("duration", {}).get("text", "")
				steps.append(f"{inst} ({dist}, {dur})")
		return "\n".join(steps) if steps else "No steps available."
	except Exception as e:
		raise ToolError(str(e))


@tool
def google_distance_matrix(origins: List[str], destinations: List[str], mode: str = "driving") -> str:
	"""Get travel time and distance between multiple origins and destinations using Google Distance Matrix API."""
	settings = load_settings()
	api_key = settings.google_maps_api_key
	if not api_key:
		raise ToolError("Missing GOOGLE_MAPS_API_KEY")
	try:
		params = {
			"origins": "|".join(origins),
			"destinations": "|".join(destinations),
			"mode": mode,
			"key": api_key,
		}
		resp = httpx.get("https://maps.googleapis.com/maps/api/distancematrix/json", params=params, timeout=20.0)
		resp.raise_for_status()
		j = resp.json()
		rows = j.get("rows", [])
		if not rows:
			return "No results."
		lines = []
		for i, row in enumerate(rows):
			for jdx, elem in enumerate(row.get("elements", [])):
				status = elem.get("status")
				if status != "OK":
					lines.append(f"{origins[i]} -> {destinations[jdx]}: {status}")
					continue
				dist = elem.get("distance", {}).get("text", "")
				dur = elem.get("duration", {}).get("text", "")
				lines.append(f"{origins[i]} -> {destinations[jdx]}: {dist}, {dur}")
		return "\n".join(lines)
	except Exception as e:
		raise ToolError(str(e))


@tool
def google_validate_address(address: str) -> str:
	"""Validate a postal address using Google Address Validation API and return normalized formatting."""
	settings = load_settings()
	api_key = settings.google_maps_api_key
	if not api_key:
		raise ToolError("Missing GOOGLE_MAPS_API_KEY")
	try:
		resp = httpx.post(
			f"https://addressvalidation.googleapis.com/v1:validateAddress?key={api_key}",
			json={"address": {"addressLines": [address]}},
			timeout=20.0,
		)
		resp.raise_for_status()
		j = resp.json()
		r = j.get("result", {})
		verdict = r.get("verdict", {})
		addr = r.get("address", {})
		lines = addr.get("formattedAddress", "") or " ".join(addr.get("addressLines", []))
		is_valid = verdict.get("hasInferredComponents", False) or verdict.get("hasReplacedComponents", False) or verdict.get("addressComplete", False)
		return f"Valid: {bool(is_valid)}\n{lines}"
	except Exception as e:
		raise ToolError(str(e))


@tool
def google_places_aggregate(query: str, near: Optional[str] = None, radius_m: int = 1000, limit: int = 10) -> str:
	"""Search places with Google Places (v1 places:searchText). Optionally bias by a location.

	Args:
	- query: free text, e.g., 'best coffee', 'pharmacy', 'EV charging'.
	- near: optional location string to bias results (geocoded if provided).
	- radius_m: bias radius in meters (default 1000).
	- limit: max results to include (default 10).

	Returns newline list: 'Name — rating(reviews) — address — primaryType'.
	"""
	settings = load_settings()
	api_key = settings.google_maps_api_key
	if not api_key:
		raise ToolError("Missing GOOGLE_MAPS_API_KEY")

	center_lat = None
	center_lng = None
	try:
		if near:
			geo = httpx.get(
				"https://maps.googleapis.com/maps/api/geocode/json",
				params={"address": near, "key": api_key},
				timeout=15.0,
			)
			geo.raise_for_status()
			gj = geo.json()
			res = gj.get("results", [])
			if res:
				loc = res[0]["geometry"]["location"]
				center_lat = loc.get("lat")
				center_lng = loc.get("lng")

		body: Dict[str, Any] = {"textQuery": query}
		if center_lat is not None and center_lng is not None:
			body["locationBias"] = {
				"circle": {
					"center": {"latitude": center_lat, "longitude": center_lng},
					"radius": int(radius_m),
				}
			}

		headers = {
			"X-Goog-Api-Key": api_key,
			"X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount,places.primaryType,places.googleMapsUri",
		}
		resp = httpx.post(
			"https://places.googleapis.com/v1/places:searchText",
			json=body,
			headers=headers,
			timeout=20.0,
		)
		resp.raise_for_status()
		pj = resp.json()
		places = pj.get("places", [])[: max(1, int(limit))]
		if not places:
			return "No places found."
		lines = []
		for p in places:
			name = (p.get("displayName", {}) or {}).get("text", "(unnamed)")
			addr = p.get("formattedAddress", "")
			rating = p.get("rating")
			reviews = p.get("userRatingCount")
			ptype = p.get("primaryType", "")
			line = f"{name} — {rating or '?'}({reviews or 0}) — {addr} — {ptype}"
			lines.append(line)
		return "\n".join(lines)
	except Exception as e:
		raise ToolError(str(e))


