#!/usr/bin/env python3
"""
Extract supervisor entrypoint script from the installed package.

This utility extracts the supervisor-entrypoint.sh script from the installed
package to a specified location, making it easy to use in Docker containers.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    import pkg_resources  # type: ignore
except ImportError:
    print("ERROR: pkg_resources not available. Install setuptools.", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    """Main entry point for the script extraction utility."""
    parser = argparse.ArgumentParser(
        description="Extract supervisor-entrypoint.sh from the installed package"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="/opt/aws/supervisor-entrypoint.sh",
        help="Output path for the entrypoint script (default: /opt/aws/supervisor-entrypoint.sh)",
    )

    parser.add_argument(
        "--make-executable",
        action="store_true",
        default=True,
        help="Make the extracted script executable (default: true)",
    )

    args = parser.parse_args()

    try:
        # Get the script path from the installed package
        script_path = pkg_resources.resource_filename(
            "model_hosting_container_standards",
            "supervisor/scripts/supervisor-entrypoint.sh",
        )

        if not os.path.exists(script_path):
            print(f"ERROR: Script not found at {script_path}", file=sys.stderr)
            return 1

        # Create output directory if it doesn't exist
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the script
        shutil.copy2(script_path, args.output)

        # Make executable if requested
        if args.make_executable:
            os.chmod(args.output, 0o755)

        print(f"Successfully extracted supervisor-entrypoint.sh to {args.output}")
        return 0

    except Exception as e:
        print(f"ERROR: Failed to extract script: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
