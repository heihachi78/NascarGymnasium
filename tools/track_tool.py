#!/usr/bin/env python3
"""
Track Tool CLI

Command line interface for track analysis, validation, and visualization.
Provides batch processing capabilities and report generation.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.track_analyzer import TrackAnalyzer, TrackStatistics
from tools.track_validator import TrackValidator, ValidationResult
from src.track_generator import TrackLoader


def analyze_single_track(file_path: str, analyzer: TrackAnalyzer, validator: TrackValidator, 
                        verbose: bool = False) -> tuple:
    """
    Analyze a single track file.
    
    Args:
        file_path (str): Path to the track file
        analyzer (TrackAnalyzer): Track analyzer instance
        validator (TrackValidator): Track validator instance
        verbose (bool): Whether to include detailed output
        
    Returns:
        tuple: (success, stats, validation_result, error_message)
    """
    try:
        # Analyze the track
        stats = analyzer.analyze_track_file(file_path)
        validation = validator.validate_track(analyzer.loader.load_track(file_path))
        
        return True, stats, validation, None
        
    except Exception as e:
        return False, None, None, str(e)


def format_validation_summary(validation: ValidationResult) -> str:
    """Format validation results as a summary string."""
    if validation.is_valid:
        status = "✓ VALID"
    else:
        status = f"✗ INVALID ({len(validation.errors)} errors)"
    
    if validation.warnings:
        status += f", {len(validation.warnings)} warnings"
    
    return status


def generate_json_report(stats: TrackStatistics, validation: ValidationResult, 
                        track_name: str) -> dict:
    """Generate a JSON report for a track."""
    return {
        "track_name": track_name,
        "analysis": {
            "total_length": stats.total_length,
            "segment_count": stats.segment_count,
            "curve_count": stats.curve_count,
            "straight_count": stats.straight_count,
            "average_width": stats.average_width,
            "track_area": stats.track_area,
            "estimated_lap_time": stats.estimated_lap_time,
            "technical_difficulty": stats.technical_difficulty,
            "bounds": {
                "min": stats.track_bounds[0],
                "max": stats.track_bounds[1]
            }
        },
        "validation": {
            "is_valid": validation.is_valid,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "suggestions": validation.suggestions
        }
    }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Track analysis and validation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze track.track                    # Analyze single track
  %(prog)s analyze tracks/*.track --batch         # Batch analyze all tracks
  %(prog)s validate track.track                   # Validate single track
  %(prog)s gui                                    # Launch GUI
  %(prog)s analyze track.track --json out.json    # Generate JSON report
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze track files')
    analyze_parser.add_argument('files', nargs='+', help='Track files to analyze')
    analyze_parser.add_argument('--batch', action='store_true', 
                               help='Process multiple files in batch mode')
    analyze_parser.add_argument('--json', metavar='FILE', 
                               help='Output JSON report to specified file')
    analyze_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Show detailed analysis')
    analyze_parser.add_argument('--quiet', '-q', action='store_true',
                               help='Only show summary information')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate track files')
    validate_parser.add_argument('files', nargs='+', help='Track files to validate')
    validate_parser.add_argument('--strict', action='store_true',
                                help='Treat warnings as errors')
    validate_parser.add_argument('--batch', action='store_true',
                                help='Process multiple files in batch mode')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch graphical interface')
    gui_parser.add_argument('file', nargs='?', help='Optional track file to load')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available track files')
    list_parser.add_argument('--directory', '-d', default='tracks',
                            help='Directory to search for track files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize tools
    analyzer = TrackAnalyzer()
    validator = TrackValidator()
    
    if args.command == 'analyze':
        return handle_analyze_command(args, analyzer, validator)
    elif args.command == 'validate':
        return handle_validate_command(args, validator)
    elif args.command == 'gui':
        return handle_gui_command(args)
    elif args.command == 'list':
        return handle_list_command(args)
    
    return 0


def handle_analyze_command(args, analyzer: TrackAnalyzer, validator: TrackValidator) -> int:
    """Handle the analyze command."""
    results = []
    failed_files = []
    
    for file_pattern in args.files:
        # Handle glob patterns
        if '*' in file_pattern:
            import glob
            files = glob.glob(file_pattern)
            if not files:
                print(f"No files found matching pattern: {file_pattern}")
                continue
        else:
            files = [file_pattern]
        
        for file_path in files:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                failed_files.append(file_path)
                continue
            
            track_name = os.path.basename(file_path)
            
            if args.batch and not args.verbose:
                print(f"Analyzing {track_name}...", end=' ')
            elif not args.quiet:
                print(f"\nAnalyzing: {track_name}")
                print("-" * 40)
            
            success, stats, validation, error = analyze_single_track(
                file_path, analyzer, validator, args.verbose
            )
            
            if not success:
                print(f"✗ FAILED: {error}")
                failed_files.append(file_path)
                continue
            
            if args.batch and not args.verbose:
                # Brief summary for batch mode
                status = "✓" if validation.is_valid else "✗"
                print(f"{status} {stats.total_length:.0f}m, {stats.segment_count} segments")
            elif not args.quiet:
                # Detailed output
                if args.verbose:
                    print(analyzer.generate_report(stats, track_name))
                else:
                    # Standard output
                    print(f"Length: {stats.total_length:.1f}m")
                    print(f"Segments: {stats.segment_count} ({stats.curve_count} curves)")
                    print(f"Width: {stats.average_width:.1f}m")
                    print(f"Estimated lap time: {stats.estimated_lap_time:.1f}s")
                    print(f"Validation: {format_validation_summary(validation)}")
                    
                    if validation.errors:
                        print("\nErrors:")
                        for error in validation.errors:
                            print(f"  • {error}")
                    
                    if validation.warnings and args.verbose:
                        print("\nWarnings:")
                        for warning in validation.warnings:
                            print(f"  • {warning}")
            
            # Store results for JSON output
            if args.json:
                results.append(generate_json_report(stats, validation, track_name))
    
    # Generate JSON report if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nJSON report saved to: {args.json}")
    
    # Summary
    if args.batch or len(args.files) > 1:
        import glob
        total_files = sum(len([f]) if '*' not in pattern else len(glob.glob(pattern)) 
                         for pattern in args.files for f in [pattern])
        successful = total_files - len(failed_files)
        print(f"\nSummary: {successful}/{total_files} files processed successfully")
    
    return 1 if failed_files else 0


def handle_validate_command(args, validator: TrackValidator) -> int:
    """Handle the validate command."""
    loader = TrackLoader()
    failed_files = []
    invalid_files = []
    
    for file_pattern in args.files:
        # Handle glob patterns
        if '*' in file_pattern:
            import glob
            files = glob.glob(file_pattern)
            if not files:
                print(f"No files found matching pattern: {file_pattern}")
                continue
        else:
            files = [file_pattern]
        
        for file_path in files:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                failed_files.append(file_path)
                continue
            
            track_name = os.path.basename(file_path)
            
            try:
                track = loader.load_track(file_path)
                validation = validator.validate_track(track)
                
                # Determine if file passes validation
                has_errors = len(validation.errors) > 0
                has_warnings = len(validation.warnings) > 0
                fails_validation = has_errors or (args.strict and has_warnings)
                
                if args.batch:
                    # Brief output for batch mode
                    if fails_validation:
                        status = "✗ FAIL"
                        invalid_files.append(file_path)
                    else:
                        status = "✓ PASS"
                    
                    warnings_text = f", {len(validation.warnings)} warnings" if has_warnings else ""
                    print(f"{track_name}: {status}{warnings_text}")
                else:
                    # Detailed output
                    print(f"\nValidating: {track_name}")
                    print("-" * 40)
                    
                    if validation.is_valid and not (args.strict and has_warnings):
                        print("✓ VALIDATION PASSED")
                    else:
                        print("✗ VALIDATION FAILED")
                        invalid_files.append(file_path)
                    
                    if validation.errors:
                        print(f"\nErrors ({len(validation.errors)}):")
                        for error in validation.errors:
                            print(f"  • {error}")
                    
                    if validation.warnings:
                        print(f"\nWarnings ({len(validation.warnings)}):")
                        for warning in validation.warnings:
                            print(f"  • {warning}")
                    
                    if validation.suggestions:
                        print(f"\nSuggestions ({len(validation.suggestions)}):")
                        for suggestion in validation.suggestions:
                            print(f"  • {suggestion}")
                
            except Exception as e:
                print(f"✗ ERROR processing {track_name}: {e}")
                failed_files.append(file_path)
    
    # Summary
    if args.batch or len(args.files) > 1:
        import glob
        total_files = sum(len([f]) if '*' not in pattern else len(glob.glob(pattern)) 
                         for pattern in args.files for f in [pattern])
        processed = total_files - len(failed_files)
        valid = processed - len(invalid_files)
        print(f"\nSummary: {valid}/{processed} files passed validation")
    
    return 1 if failed_files or invalid_files else 0


def handle_gui_command(args) -> int:
    """Handle the GUI command."""
    try:
        from track_builder import TrackBuilderGUI
        
        app = TrackBuilderGUI()
        
        # Load specified file if provided
        if args.file:
            if os.path.exists(args.file):
                app.load_track_file(args.file)
            else:
                print(f"Warning: File not found: {args.file}")
        
        app.run()
        return 0
        
    except ImportError as e:
        print(f"GUI not available: {e}")
        print("Make sure pygame is installed: pip install pygame")
        return 1
    except Exception as e:
        print(f"Error launching GUI: {e}")
        return 1


def handle_list_command(args) -> int:
    """Handle the list command."""
    directory = args.directory
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return 1
    
    track_files = []
    for file in os.listdir(directory):
        if file.endswith('.track'):
            track_files.append(file)
    
    if not track_files:
        print(f"No track files found in {directory}")
        return 0
    
    print(f"Track files in {directory}:")
    print("-" * 40)
    
    for track_file in sorted(track_files):
        file_path = os.path.join(directory, track_file)
        try:
            # Quick analysis for file size and basic info
            file_size = os.path.getsize(file_path)
            
            # Count lines for rough complexity estimate
            with open(file_path, 'r') as f:
                line_count = sum(1 for line in f if line.strip() and not line.strip().startswith('#'))
            
            print(f"  {track_file:<20} ({file_size:>4} bytes, {line_count:>2} commands)")
            
        except Exception as e:
            print(f"  {track_file:<20} (error: {e})")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())