#!/usr/bin/env python3
"""
Log Parser for vLLM output
Separates actual errors from INFO logs that vLLM writes to stderr
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class VLLMLogParser:
    """Parse vLLM stdout and stderr logs to extract structured information."""
    
    # Log level patterns
    INFO_PATTERN = re.compile(r'^\(.*?\)\s+(INFO|DEBUG)\s+', re.IGNORECASE)
    WARNING_PATTERN = re.compile(r'^\(.*?\)\s+WARNING\s+', re.IGNORECASE)
    ERROR_PATTERN = re.compile(r'^\(.*?\)\s+ERROR\s+', re.IGNORECASE)
    
    # Success indicators (these appear in logs even when successful)
    SUCCESS_INDICATORS = [
        re.compile(r'GPU KV cache size:\s*([\d,]+)\s*tokens', re.IGNORECASE),
        re.compile(r'Maximum concurrency for\s+([\d,]+)\s*tokens:\s*([\d.]+)x', re.IGNORECASE),
        re.compile(r'Using\s+(\w+)\s+backend', re.IGNORECASE),
        re.compile(r'vLLM API server.*running', re.IGNORECASE),
        re.compile(r'Uvicorn running on', re.IGNORECASE),
    ]
    
    # Error indicators
    ERROR_INDICATORS = [
        re.compile(r'Traceback\s*\(most recent call', re.IGNORECASE),
        re.compile(r'Exception:', re.IGNORECASE),
        re.compile(r'Error:', re.IGNORECASE),
        re.compile(r'FATAL:', re.IGNORECASE),
        re.compile(r'CUDA error', re.IGNORECASE),
        re.compile(r'out of memory', re.IGNORECASE),
        re.compile(r'failed to', re.IGNORECASE),
        re.compile(r'cannot import', re.IGNORECASE),
        re.compile(r'ModuleNotFoundError', re.IGNORECASE),
        re.compile(r'RuntimeError', re.IGNORECASE),
        re.compile(r'ValueError', re.IGNORECASE),
        re.compile(r'KeyError', re.IGNORECASE),
        re.compile(r'AttributeError', re.IGNORECASE),
    ]
    
    # Backend detection
    BACKEND_PATTERNS = [
        re.compile(r'Using\s+(FlashAttention|flash_attn)\s+backend', re.IGNORECASE),
        re.compile(r'Using\s+(FlashInfer|flashinfer)\s+backend', re.IGNORECASE),
        re.compile(r'Using\s+(XFormers|xformers)\s+backend', re.IGNORECASE),
        re.compile(r'Using\s+(TorchSDPA|torch_sdpa)\s+backend', re.IGNORECASE),
    ]
    
    # Hardware info
    HARDWARE_PATTERNS = {
        'cuda_version': re.compile(r'CUDA Version:\s*([\d.]+)', re.IGNORECASE),
        'gpu_name': re.compile(r'GPU.*:\s*(NVIDIA [\w\s]+)', re.IGNORECASE),
        'gpu_memory': re.compile(r'(\d+)\s*MiB', re.IGNORECASE),
        'compute_capability': re.compile(r'Compute Capability:\s*([\d.]+)', re.IGNORECASE),
    }
    
    # KV cache stats
    KV_CACHE_PATTERNS = {
        'kv_cache_tokens': re.compile(r'GPU KV cache size:\s*([\d,]+)\s*tokens', re.IGNORECASE),
        'max_concurrency': re.compile(r'Maximum concurrency for\s+([\d,]+)\s*tokens:\s*([\d.]+)x', re.IGNORECASE),
        'gpu_memory_utilization': re.compile(r'gpu-memory-utilization\s*=\s*([\d.]+)', re.IGNORECASE),
    }
    
    @classmethod
    def parse_logs(cls, stdout_path: Path, stderr_path: Path) -> Dict:
        """
        Parse both stdout and stderr logs.
        Returns structured information about the run.
        """
        result = {
            'log_levels': {'info': 0, 'warning': 0, 'error': 0, 'unknown': 0},
            'success_indicators': [],
            'errors': [],
            'warnings': [],
            'backends': {'detected': [], 'used': None},
            'hardware': {},
            'kv_cache': {},
            'error_classification': None,
            'likely_success': False,
        }
        
        # Read both files
        lines = []
        for path in [stdout_path, stderr_path]:
            if path.exists():
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines.extend(f.readlines())
        
        # Parse each line
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Classify log level
            log_level = cls._classify_log_level(line)
            result['log_levels'][log_level] += 1
            
            # Check for success indicators
            for pattern in cls.SUCCESS_INDICATORS:
                match = pattern.search(line)
                if match:
                    result['success_indicators'].append({
                        'pattern': pattern.pattern[:50],
                        'match': match.group(0),
                        'groups': match.groups()
                    })
            
            # Check for errors (only ERROR level, not INFO even if it looks scary)
            if log_level == 'error':
                error_info = {'line': line[:200]}
                for pattern in cls.ERROR_INDICATORS:
                    if pattern.search(line):
                        error_info['type'] = pattern.pattern[:50]
                        break
                result['errors'].append(error_info)
            
            # Check for warnings
            if log_level == 'warning':
                result['warnings'].append(line[:200])
            
            # Extract backends
            for pattern in cls.BACKEND_PATTERNS:
                match = pattern.search(line)
                if match:
                    backend = match.group(1)
                    if backend not in result['backends']['detected']:
                        result['backends']['detected'].append(backend)
                    result['backends']['used'] = backend
            
            # Extract hardware info
            for key, pattern in cls.HARDWARE_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    result['hardware'][key] = match.group(1)
            
            # Extract KV cache stats
            for key, pattern in cls.KV_CACHE_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    if key == 'max_concurrency':
                        result['kv_cache']['target_tokens'] = int(match.group(1).replace(',', ''))
                        result['kv_cache']['concurrency_factor'] = float(match.group(2))
                    elif key == 'kv_cache_tokens':
                        result['kv_cache']['total_tokens'] = int(match.group(1).replace(',', ''))
                    else:
                        result['kv_cache'][key] = match.group(1)
        
        # Determine if likely successful
        # Success = has KV cache stats OR has backend detection AND no actual errors
        has_kv_stats = bool(result['kv_cache'].get('total_tokens'))
        has_backend = bool(result['backends']['used'])
        has_real_errors = len(result['errors']) > 0
        
        result['likely_success'] = (has_kv_stats or has_backend) and not has_real_errors
        
        # Classify error if failed
        if has_real_errors and result['errors']:
            result['error_classification'] = cls._classify_error(result['errors'])
        
        return result
    
    @classmethod
    def _classify_log_level(cls, line: str) -> str:
        """Classify the log level of a line."""
        if cls.ERROR_PATTERN.search(line):
            return 'error'
        elif cls.WARNING_PATTERN.search(line):
            return 'warning'
        elif cls.INFO_PATTERN.search(line):
            return 'info'
        else:
            return 'unknown'
    
    @classmethod
    def _classify_error(cls, errors: List[Dict]) -> Dict:
        """Classify the type of error."""
        error_text = ' '.join([e['line'] for e in errors]).lower()
        
        classifications = [
            ('backend', ['backend', 'flash_attn', 'flashinfer', 'xformers', 'cutlass', 'kernel']),
            ('oom', ['out of memory', 'oom', 'cuda oom', 'allocation failed', 'insufficient memory']),
            ('fp8', ['fp8', 'illegal memory access', 'dtype']),
            ('prefill', ['prefill', 'chunked_prefill', 'attention mask']),
            ('trust_remote_code', ['trust_remote_code', 'custom architecture']),
            ('cuda_graph', ['cuda graph', 'graph capture', 'cudagraph']),
            ('tensor_parallel', ['tensor parallel', 'distributed', 'nccl']),
            ('import', ['cannot import', 'modulenotfound', 'importerror']),
        ]
        
        for error_type, keywords in classifications:
            if any(kw in error_text for kw in keywords):
                return {
                    'type': error_type,
                    'confidence': 'high',
                    'keywords_found': [kw for kw in keywords if kw in error_text][:3]
                }
        
        return {
            'type': 'unknown',
            'confidence': 'low',
            'keywords_found': []
        }


def parse_vllm_logs(stdout_path: str, stderr_path: str) -> Dict:
    """Convenience function to parse vLLM logs."""
    return VLLMLogParser.parse_logs(Path(stdout_path), Path(stderr_path))


if __name__ == "__main__":
    import json
    import sys
    
    if len(sys.argv) >= 3:
        stdout = Path(sys.argv[1])
        stderr = Path(sys.argv[2])
        result = parse_vllm_logs(stdout, stderr)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python log_parser.py <stdout.log> <stderr.log>")
