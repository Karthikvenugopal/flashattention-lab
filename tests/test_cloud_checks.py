import xml.etree.ElementTree as ET
from unittest.mock import Mock
import pytest
from attention_lab.cloud_checks import run_verified_benchmark


def junit(count=16, issue=None):
    root = ET.Element('testsuite')
    for i in range(count):
        case = ET.SubElement(root, 'testcase', name=f'test_fused_reference[{i}]')
        if i == 0 and issue:
            ET.SubElement(case, issue)
    return ET.tostring(root)


@pytest.mark.parametrize('returncode,report', [(1, junit()), (5, junit()), (0, b''),
    (0, b'<testsuite/>'), (0, junit(15)), (0, junit(issue='failure')),
    (0, junit(issue='error')), (0, junit(issue='skipped'))])
def test_benchmarks_cannot_start_when_correctness_is_unproven(returncode, report):
    benchmark = Mock()
    with pytest.raises(ValueError):
        run_verified_benchmark(returncode, report, benchmark)
    benchmark.assert_not_called()


def test_all_gpu_cases_pass_before_benchmark_called_once():
    benchmark = Mock(return_value={'status': 'passed'})
    assert run_verified_benchmark(0, junit(), benchmark) == {'status': 'passed'}
    benchmark.assert_called_once_with()
