#!/usr/bin/env python3
"""
Load environment variables from a sectioned .env file to a conda environment.
Supports platform-specific sections and command line arguments.
"""

import subprocess
import os
import sys
import argparse
import platform
from pathlib import Path


def detect_machine_name():
    """
    Auto-detect machine name based on hostname and platform.
    You can customize this function to match your machine naming conventions.
    """
    hostname = platform.node().lower()
    system = platform.system().lower()
    
    # Common cluster patterns - customize these for your setup
    if 'login' in hostname or 'head' in hostname:
        return 'cluster'
    elif 'hpc' in hostname or 'compute' in hostname:
        return 'cluster'
    elif system == 'linux' and 'ubuntu' in platform.platform().lower():
        return 'linux'
    elif system == 'darwin':
        return 'mac'
    elif system == 'windows':
        return 'windows'
    else:
        return 'default'


def parse_env_file(env_file, machine_name):
    """
    Parse environment file with sections.
    
    Expected format:
    # Global variables (always loaded)
    GLOBAL_VAR=value
    
    [default]
    DEFAULT_VAR=default_value
    
    [linux]
    GDAL_DATA=/usr/share/gdal
    PROJ_LIB=/usr/share/proj
    
    [mac]
    GDAL_DATA=/opt/homebrew/share/gdal
    PROJ_LIB=/opt/homebrew/share/proj
    
    [cluster]
    SCRATCH_DIR=/scratch/username
    MODULE_PATH=/opt/modules
    """
    env_path = Path(env_file)
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file {env_file} not found")
    
    variables = {}
    current_section = None
    
    print(f"Parsing {env_file} for machine: {machine_name}")
    
    with open(env_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check for section headers [section_name]
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].strip().lower()
                print(f"Found section: [{current_section}]")
                continue
            
            # Parse key=value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                
                # Load global variables (no section) or variables from matching section
                if current_section is None or current_section == machine_name.lower():
                    variables[key] = value
                    section_label = f"[{current_section}]" if current_section else "[global]"
                    print(f"  {section_label} {key}={value}")
                else:
                    print(f"  [{current_section}] {key}={value} (skipped)")
            else:
                print(f"Warning: Line {line_num} has invalid format: {line}")
    
    return variables


def set_conda_variables(variables, conda_env):
    """Set variables in the specified conda environment."""
    if not variables:
        print("No variables to set.")
        return
    
    print(f"\nSetting {len(variables)} variables in conda environment '{conda_env}':")
    
    success_count = 0
    for key, value in variables.items():
        try:
            cmd = f'conda env config vars set -n {conda_env} {key}={value}'
            result = subprocess.run(cmd, shell=True, check=True, 
                                  capture_output=True, text=True)
            print(f"  ✓ {key}={value}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to set {key}: {e.stderr.strip()}")
    
    print(f"\nSuccessfully set {success_count}/{len(variables)} variables.")
    if success_count > 0:
        print("Restart your Jupyter kernel to apply changes.")


def main():
    parser = argparse.ArgumentParser(
        description="Load environment variables from sectioned file to conda environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example .env file format:
  # Global variables
  PYTHONPATH=/common/path
  
  [linux]
  GDAL_DATA=/usr/share/gdal
  PROJ_LIB=/usr/share/proj
  
  [mac]
  GDAL_DATA=/opt/homebrew/share/gdal
  
  [cluster]
  SCRATCH_DIR=/scratch/username
  SLURM_CONF=/etc/slurm/slurm.conf

Example usage:
  python load_env_to_conda.py
  python load_env_to_conda.py --env-file config.env --conda-env myenv --machine-name cluster
        """
    )
    
    # Auto-detect defaults
    default_machine = detect_machine_name()
    
    parser.add_argument(
        '--env-file', 
        default='.env',
        help='Path to environment file (default: .env)'
    )
    parser.add_argument(
        '--conda-env',
        default=os.environ.get('CONDA_DEFAULT_ENV', 'base'),
        help='Conda environment name (default: current active environment)'
    )
    parser.add_argument(
        '--machine-name',
        default=default_machine,
        help=f'Machine/platform name for section selection (default: auto-detected "{default_machine}")'
    )
    parser.add_argument(
        '--list-sections',
        action='store_true',
        help='List all sections in the env file and exit'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be set without actually setting variables'
    )
    
    args = parser.parse_args()
    
    try:
        # List sections mode
        if args.list_sections:
            env_path = Path(args.env_file)
            if not env_path.exists():
                print(f"Error: {args.env_file} not found")
                sys.exit(1)
            
            sections = set()
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        sections.add(line[1:-1])
            
            print(f"Sections found in {args.env_file}:")
            for section in sorted(sections):
                print(f"  - {section}")
            if not sections:
                print("  (no sections found - file contains global variables only)")
            sys.exit(0)
        
        # Parse environment variables
        variables = parse_env_file(args.env_file, args.machine_name)
        
        if not variables:
            print(f"No variables found for machine '{args.machine_name}' in {args.env_file}")
            print("Use --list-sections to see available sections.")
            sys.exit(1)
        
        # Dry run mode
        if args.dry_run:
            print(f"\nDRY RUN: Would set these variables in '{args.conda_env}':")
            for key, value in variables.items():
                print(f"  {key}={value}")
            print(f"\nRun without --dry-run to actually set {len(variables)} variables.")
        else:
            # Actually set the variables
            set_conda_variables(variables, args.conda_env)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()