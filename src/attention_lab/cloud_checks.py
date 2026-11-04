"""Fail-closed prerequisite for paid NVIDIA benchmarks; no cloud imports."""
import xml.etree.ElementTree as ET


def run_verified_benchmark(returncode, junit_text, benchmark):
    """Require all 16 fused GPU cases to execute and pass before calling benchmark."""
    if returncode != 0:
        raise ValueError(f'Triton pytest exited with status {returncode}')
    try:
        root = ET.fromstring(junit_text)
    except (ET.ParseError, TypeError) as exc:
        raise ValueError('Missing or invalid Triton JUnit report') from exc
    cases = root.findall('.//testcase')
    if not cases or any(case.find(tag) is not None for case in cases for tag in ('failure', 'error', 'skipped')):
        raise ValueError('Triton test results contain failures/errors/skips or no tests')
    if root.findall('.//error') or root.findall('.//failure'):
        raise ValueError('Triton test session reported an error or failure')
    fused = [case for case in cases if case.get('name', '').startswith('test_fused_reference[')]
    if len(fused) != 16 or len({case.get('name') for case in fused}) != 16:
        raise ValueError('Expected 16 distinct, executed fused GPU correctness cases')
    return benchmark()
