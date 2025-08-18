#!/usr/bin/env python3
"""
Run metadata tracking system for scikit-maad GUI

Captures comprehensive information about each analysis run including:
- User settings and parameters
- File paths and processing details
- System information and performance metrics
- Results and output locations
"""

import json
import os
import time
import datetime
import platform
import getpass
from pathlib import Path
import hashlib

class RunMetadata:
    """Tracks and saves metadata for each analysis run"""
    
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.metadata = {
            'run_info': {},
            'system_info': {},
            'input_settings': {},
            'processing_info': {},
            'output_info': {},
            'performance_metrics': {},
            'file_manifest': {}
        }
        self.start_time = None
        self.end_time = None
        
    def start_run(self, **kwargs):
        """Initialize run metadata at the start of processing"""
        self.start_time = time.time()
        
        # Basic run information
        self.metadata['run_info'] = {
            'run_id': self._generate_run_id(),
            'start_time': datetime.datetime.now().isoformat(),
            'user': getpass.getuser(),
            'gui_version': 'Marine Acoustics with Parallel Processing v1.0',
            'working_directory': os.getcwd()
        }
        
        # System information
        self.metadata['system_info'] = {
            'platform': platform.platform(),
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count()
        }
        
        # Input settings from GUI
        self.metadata['input_settings'] = kwargs
        
    def add_processing_info(self, **kwargs):
        """Add information about the processing phase"""
        self.metadata['processing_info'].update(kwargs)
        
    def add_file_info(self, input_files, output_files):
        """Add information about input and output files"""
        self.metadata['file_manifest'] = {
            'input_files': {
                'count': len(input_files),
                'files': [{'path': f, 'size': self._get_file_size(f)} for f in input_files],
                'total_size_mb': sum(self._get_file_size(f) for f in input_files) / (1024*1024)
            },
            'output_files': {
                'count': len(output_files),
                'files': output_files,
                'output_folder': self.output_folder
            }
        }
        
    def add_performance_metrics(self, **kwargs):
        """Add performance timing and resource usage"""
        self.metadata['performance_metrics'].update(kwargs)
        
    def finish_run(self, success=True, error_message=None):
        """Finalize run metadata at the end of processing"""
        self.end_time = time.time()
        
        self.metadata['run_info'].update({
            'end_time': datetime.datetime.now().isoformat(),
            'duration_seconds': self.end_time - self.start_time if self.start_time else None,
            'success': success,
            'error_message': error_message
        })
        
        # Save metadata to file
        self._save_metadata()
        
    def _generate_run_id(self):
        """Generate unique run ID"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"
        
    def _get_file_size(self, filepath):
        """Get file size in bytes, return 0 if file doesn't exist"""
        try:
            return os.path.getsize(filepath)
        except (OSError, FileNotFoundError):
            return 0
            
    def _save_metadata(self):
        """Save metadata to JSON file in output folder"""
        try:
            # Create output folder if it doesn't exist
            os.makedirs(self.output_folder, exist_ok=True)
            
            # Generate filename using run_identifier and timestamp
            run_identifier = self.metadata['input_settings'].get('run_identifier', '').strip()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if run_identifier:
                filename = f"metadata_{run_identifier}_{timestamp}.json"
            else:
                filename = f"metadata_{timestamp}.json"
            filepath = os.path.join(self.output_folder, filename)
            
            # Save metadata
            with open(filepath, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
                
            print(f"📋 Run metadata saved: {filepath}")
            
            # Also create a human-readable summary
            self._save_summary_report(filepath.replace('.json', '_summary.txt'))
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save run metadata: {e}")
            
    def _save_summary_report(self, filepath):
        """Save human-readable summary report"""
        try:
            with open(filepath, 'w') as f:
                f.write("SCIKIT-MAAD ANALYSIS RUN SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                # Run information
                run_info = self.metadata['run_info']
                f.write(f"Run ID: {run_info.get('run_id', 'N/A')}\n")
                f.write(f"Date/Time: {run_info.get('start_time', 'N/A')}\n")
                f.write(f"User: {run_info.get('user', 'N/A')}\n")
                f.write(f"Duration: {run_info.get('duration_seconds', 0):.1f} seconds\n")
                f.write(f"Success: {'Yes' if run_info.get('success') else 'No'}\n")
                if run_info.get('error_message'):
                    f.write(f"Error: {run_info['error_message']}\n")
                f.write("\n")
                
                # System information
                sys_info = self.metadata['system_info']
                f.write("SYSTEM INFORMATION\n")
                f.write("-" * 20 + "\n")
                f.write(f"Platform: {sys_info.get('platform', 'N/A')}\n")
                f.write(f"CPU Cores: {sys_info.get('cpu_count', 'N/A')}\n")
                f.write(f"Python Version: {sys_info.get('python_version', 'N/A')}\n")
                f.write("\n")
                
                # Input settings
                input_settings = self.metadata['input_settings']
                f.write("INPUT SETTINGS\n")
                f.write("-" * 15 + "\n")
                f.write(f"Input Folder: {input_settings.get('input_folder', 'N/A')}\n")
                f.write(f"Output Folder: {input_settings.get('output_folder', 'N/A')}\n")
                f.write(f"Processing Mode: {input_settings.get('mode', 'N/A')}\n")
                f.write(f"Frequency Bands: Anthro {input_settings.get('flim_low', 'N/A')}, Bio {input_settings.get('flim_mid', 'N/A')}\n")
                f.write(f"Parallel Processing: {'Yes' if input_settings.get('parallel_enabled') else 'No'}\n")
                if input_settings.get('marine_indices_enabled'):
                    f.write("Marine Indices: Enabled\n")
                f.write("\n")
                
                # File information
                file_info = self.metadata['file_manifest']
                if file_info:
                    f.write("FILE INFORMATION\n")
                    f.write("-" * 17 + "\n")
                    input_files = file_info.get('input_files', {})
                    f.write(f"Input Files: {input_files.get('count', 0)}\n")
                    f.write(f"Total Input Size: {input_files.get('total_size_mb', 0):.1f} MB\n")
                    
                    output_files = file_info.get('output_files', {})
                    f.write(f"Output Files: {output_files.get('count', 0)}\n")
                    f.write("\n")
                
                # Performance metrics
                perf_info = self.metadata['performance_metrics']
                if perf_info:
                    f.write("PERFORMANCE METRICS\n")
                    f.write("-" * 20 + "\n")
                    if 'sequential_time' in perf_info and 'parallel_time' in perf_info:
                        f.write(f"Sequential Time: {perf_info['sequential_time']:.1f}s\n")
                        f.write(f"Parallel Time: {perf_info['parallel_time']:.1f}s\n")
                        speedup = perf_info['sequential_time'] / perf_info['parallel_time']
                        f.write(f"Speedup: {speedup:.1f}x\n")
                    elif 'processing_time' in perf_info:
                        f.write(f"Processing Time: {perf_info['processing_time']:.1f}s\n")
                    
                    if 'files_processed' in perf_info:
                        f.write(f"Files Processed: {perf_info['files_processed']}\n")
                        f.write(f"Files Failed: {perf_info.get('files_failed', 0)}\n")
                        
            print(f"📄 Run summary saved: {filepath}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save run summary: {e}")

def create_run_metadata(output_folder):
    """Factory function to create a new RunMetadata instance"""
    return RunMetadata(output_folder)

def list_previous_runs(output_folder):
    """List all previous runs in the output folder"""
    try:
        metadata_files = []
        for filename in os.listdir(output_folder):
            if filename.startswith('run_metadata_') and filename.endswith('.json'):
                filepath = os.path.join(output_folder, filename)
                try:
                    with open(filepath, 'r') as f:
                        metadata = json.load(f)
                    metadata_files.append({
                        'filename': filename,
                        'run_id': metadata.get('run_info', {}).get('run_id'),
                        'start_time': metadata.get('run_info', {}).get('start_time'),
                        'success': metadata.get('run_info', {}).get('success'),
                        'files_processed': metadata.get('performance_metrics', {}).get('files_processed', 0)
                    })
                except:
                    continue
                    
        return sorted(metadata_files, key=lambda x: x['start_time'], reverse=True)
        
    except (OSError, FileNotFoundError):
        return []

def load_run_metadata(filepath):
    """Load metadata from a specific run file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata from {filepath}: {e}")
        return None

if __name__ == "__main__":
    # Test the metadata system
    print("Testing run metadata system...")
    
    # Create test metadata
    metadata = create_run_metadata("/tmp/test_output")
    metadata.start_run(
        input_folder="/test/input",
        output_folder="/test/output",
        mode="dataset",
        flim_low=[0, 1000],
        flim_mid=[1000, 8000],
        use_parallel=True,
        marine_indices_enabled=True
    )
    
    # Add some processing info
    metadata.add_processing_info(
        files_found=10,
        processing_mode="parallel"
    )
    
    # Add performance metrics
    metadata.add_performance_metrics(
        sequential_time=45.2,
        parallel_time=12.8,
        files_processed=10,
        files_failed=0
    )
    
    # Finish run
    metadata.finish_run(success=True)
    
    print("✓ Test completed - check /tmp/test_output for generated files")