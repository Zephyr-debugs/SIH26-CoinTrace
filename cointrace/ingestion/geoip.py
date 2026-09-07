"""
Phase 2 - GeoIP/ASN enrichment.

Wraps MaxMind's geoip2 reader. If the .mmdb files aren't present (e.g. you
haven't downloaded GeoLite2 yet, or the `geoip2` package isn't installed),
this degrades gracefully: enrichment fields are just left as None rather
than raising, so the rest of the pipeline is still fully buildable/testable
without the databases.
"""
from __future__ import annotations

import ipaddress
from functools import lru_cache

from cointrace import config

try:
    import geoip2.database
    import geoip2.errors
    _GEOIP2_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when geoip2 isn't installed
    _GEOIP2_AVAILABLE = False


class GeoEnricher:
    def __init__(self, city_db_path=config.GEOIP_CITY_DB, asn_db_path=config.GEOIP_ASN_DB):
        self.city_reader = None
        self.asn_reader = None
        self.enabled = False

        if not _GEOIP2_AVAILABLE:
            print("[geoip] 'geoip2' package not installed - enrichment disabled "
                  "(pip install geoip2 to enable).")
            return

        if city_db_path.exists():
            self.city_reader = geoip2.database.Reader(str(city_db_path))
        if asn_db_path.exists():
            self.asn_reader = geoip2.database.Reader(str(asn_db_path))

        if self.city_reader is None and self.asn_reader is None:
            print(f"[geoip] No .mmdb files found in {config.GEOIP_DIR} - "
                  "enrichment disabled. See README for download instructions.")
        else:
            self.enabled = True

    @staticmethod
    def _is_valid_public_ip(ip: str) -> bool:
        try:
            addr = ipaddress.ip_address(ip)
            return not (addr.is_private or addr.is_loopback or addr.is_reserved)
        except ValueError:
            return False

    @lru_cache(maxsize=100_000)
    def enrich(self, ip: str) -> dict:
        """Returns dict with keys: country, city, latitude, longitude,
        asn_org (all None if lookup unavailable/fails)."""
        result = {"country": None, "city": None, "latitude": None,
                  "longitude": None, "asn_org": None}

        if not self.enabled or not self._is_valid_public_ip(ip):
            return result

        if self.city_reader is not None:
            try:
                resp = self.city_reader.city(ip)
                result["country"] = resp.country.iso_code
                result["city"] = resp.city.name
                result["latitude"] = resp.location.latitude
                result["longitude"] = resp.location.longitude
            except geoip2.errors.AddressNotFoundError:
                pass

        if self.asn_reader is not None:
            try:
                resp = self.asn_reader.asn(ip)
                result["asn_org"] = resp.autonomous_system_organization
            except geoip2.errors.AddressNotFoundError:
                pass

        return result

    def close(self):
        if self.city_reader is not None:
            self.city_reader.close()
        if self.asn_reader is not None:
            self.asn_reader.close()
