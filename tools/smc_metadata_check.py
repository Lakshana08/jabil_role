"""Checks the Smart Material CDS service (ZSD_SMC_MATERIAL) against this app's
field mapping: reads $metadata through the same client/destination the app
uses, lists every entity set from SMC_OData_V4_URLs.xlsx with its properties,
and shows which real field each query source resolves to per role.

Run from the project root (uses .env like the app does):
    python -m tools.smc_metadata_check            # mapping + entity list
    python -m tools.smc_metadata_check --fields   # also print every property
    python -m tools.smc_metadata_check --sample   # also fetch 1 row + total per query source
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from app.core.config import get_settings  # noqa: E402
from app.core_capabilities._s4_apis import BY_NAME, SMC_ENTITY_SETS  # noqa: E402
from app.services.odata_metadata import get_metadata_cache  # noqa: E402
from app.services.odata_utils import extract_count, extract_rows  # noqa: E402
from app.services.s4_client import get_s4_client  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", action="store_true", help="print every property of every entity set")
    parser.add_argument("--sample", action="store_true", help="fetch one row and the total count per query source")
    args = parser.parse_args()

    service_path = get_settings().s4_smc_service_path
    print(f"Service: {service_path}")
    sets = get_metadata_cache().service(service_path)
    if sets is None:
        print("ERROR: couldn't read $metadata - check S4_BASE_URL / credentials / S4_SAP_CLIENT / S4_VERIFY_SSL.")
        return 1

    print(f"\n{len(sets)} entity sets published.\n")
    for source, names in SMC_ENTITY_SETS.items():
        print(f"[{source}]")
        for name in names:
            schema = sets.get(name)
            if schema is None:
                print(f"  {name:28} MISSING from $metadata")
                continue
            print(f"  {name:28} {len(schema.properties)} fields, key: {', '.join(schema.keys) or '-'}")
            if args.fields:
                for prop in schema.properties.values():
                    length = f"({prop.max_length})" if prop.max_length else ""
                    print(f"      {prop.name:32} {prop.type}{length}")
    unlisted = sorted(set(sets) - {n for names in SMC_ENTITY_SETS.values() for n in names})
    if unlisted:
        print(f"\nPublished but not in the spreadsheet: {', '.join(unlisted)}")

    print("\nQuery/Report sources (filter -> field):")
    problems = 0
    for query_type, entity in BY_NAME.items():
        print(f"  {query_type} -> {entity.entity_set}")
        schema = sets.get(entity.entity_set)
        checks = [(f"filter {name}", field) for name, field in entity.filters.items()]
        checks += [("key field", field) for field in entity.key_fields]
        for what, field in checks:
            if schema is None or schema.find((field,)) is None:
                problems += 1
                print(f"      {what:28} {field}  NOT FOUND")
        if args.sample:
            data = get_s4_client().get(entity.path, params={"$top": 1, "$count": "true"})
            rows = extract_rows(data)
            print(f"      total rows: {extract_count(data)}")
            if rows:
                print(f"      first row:  {rows[0]}")

    if problems:
        print(f"\n{problems} field(s) not found - fix the field names in app/core_capabilities/_s4_apis.py.")
        return 2
    print("\nEvery filter and key field exists on the live service.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
