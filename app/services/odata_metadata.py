"""Reads an OData V4 service's $metadata (CSDL XML) so callers can find each
entity set's real property names, types and lengths at runtime instead of
hard-coding them.

The Smart Material CDS views are a custom service (ZSD_SMC_MATERIAL), so a
field may be called Material, Product or MATNR depending on how the view was
modelled - and an older/newer service version can rename one. Resolving names
from $metadata turns that into a lookup, not a 400 Bad Request from S/4.

The document is fetched once per service and cached (an hour on success, a
minute on failure so a temporary outage doesn't pin the fallback in place)."""

import logging
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 3600
_FAILURE_TTL_SECONDS = 60


@dataclass(frozen=True)
class Property:
    name: str
    type: str = "Edm.String"
    max_length: int | None = None


@dataclass(frozen=True)
class EntitySetSchema:
    name: str
    properties: dict[str, Property] = field(default_factory=dict)
    keys: tuple[str, ...] = ()

    def find(self, candidates: tuple[str, ...]) -> Property | None:
        """First candidate the entity actually has - matched ignoring case and
        underscores, so "Material" also finds "MATERIAL" or "material"."""
        by_norm = {_norm(name): prop for name, prop in self.properties.items()}
        for candidate in candidates:
            prop = by_norm.get(_norm(candidate))
            if prop is not None:
                return prop
        return None


def _norm(name: str) -> str:
    return name.replace("_", "").lower()


def _local(tag: str) -> str:
    """'{http://docs.oasis-open.org/odata/ns/edm}EntityType' -> 'EntityType'."""
    return tag.rsplit("}", 1)[-1]


def parse_metadata(xml_text: str) -> dict[str, EntitySetSchema]:
    """CSDL $metadata -> {entity set name: schema}. Entity types are matched to
    their entity set by the type's short name (namespace/alias stripped)."""
    root = ET.fromstring(xml_text)

    types: dict[str, EntitySetSchema] = {}
    for element in root.iter():
        if _local(element.tag) != "EntityType":
            continue
        props: dict[str, Property] = {}
        keys: list[str] = []
        for child in element:
            kind = _local(child.tag)
            if kind == "Property":
                max_length = child.get("MaxLength")
                props[child.get("Name")] = Property(
                    name=child.get("Name"),
                    type=child.get("Type", "Edm.String"),
                    max_length=int(max_length) if max_length and max_length.isdigit() else None,
                )
            elif kind == "Key":
                keys.extend(ref.get("Name") for ref in child if _local(ref.tag) == "PropertyRef")
        types[element.get("Name")] = EntitySetSchema(name=element.get("Name"), properties=props, keys=tuple(keys))

    entity_sets: dict[str, EntitySetSchema] = {}
    for element in root.iter():
        if _local(element.tag) != "EntitySet":
            continue
        type_name = (element.get("EntityType") or "").rsplit(".", 1)[-1]
        entity_type = types.get(type_name)
        if entity_type is not None:
            entity_sets[element.get("Name")] = EntitySetSchema(
                name=element.get("Name"), properties=entity_type.properties, keys=entity_type.keys
            )
    return entity_sets


class MetadataCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[dict[str, EntitySetSchema] | None, float]] = {}

    def entity_set(self, service_path: str, entity_set: str) -> EntitySetSchema | None:
        """The entity set's schema, or None when $metadata can't be read (callers
        then fall back to their first candidate field name)."""
        sets = self.service(service_path)
        return sets.get(entity_set) if sets is not None else None

    def service(self, service_path: str) -> dict[str, EntitySetSchema] | None:
        # Held across the fetch on purpose: a burst of first requests triggers
        # one $metadata download, not one per request.
        with self._lock:
            cached = self._cache.get(service_path)
            if cached and cached[1] > time.time():
                return cached[0]
            sets = self._fetch(service_path)
            ttl = _CACHE_TTL_SECONDS if sets is not None else _FAILURE_TTL_SECONDS
            self._cache[service_path] = (sets, time.time() + ttl)
            return sets

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @staticmethod
    def _fetch(service_path: str) -> dict[str, EntitySetSchema] | None:
        from app.services.s4_client import get_s4_client  # avoid an import cycle at module load

        try:
            xml_text = get_s4_client().get_text(f"{service_path.rstrip('/')}/$metadata")
            return parse_metadata(xml_text)
        except Exception as exc:  # noqa: BLE001 - any failure just means "no metadata", logged
            logger.warning("Couldn't read $metadata for %s (%s) - using default field names", service_path, exc)
            return None


_metadata = MetadataCache()


def get_metadata_cache() -> MetadataCache:
    return _metadata
