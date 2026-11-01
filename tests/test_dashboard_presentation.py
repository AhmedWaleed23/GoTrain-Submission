from gotrain.report.html import REQUIRED_TASKS, dashboard_tasks
from gotrain.report.friendly import upgrade_document


def test_cattle_only_dashboard_does_not_invent_missing_required_tasks():
    assert dashboard_tasks([{"task": "m-nz-cattle"}]) == ["m-nz-cattle"]


def test_main_and_empty_dashboards_keep_required_task_warnings():
    assert set(dashboard_tasks([])) == set(REQUIRED_TASKS)
    assert set(dashboard_tasks([{"task": "m-eurosat"}])) == set(REQUIRED_TASKS)
    assert dashboard_tasks([{"task": "synthetic"}], fixture=True) == ["synthetic"]


def test_friendly_dashboard_shows_only_available_task_cards():
    source = """<html><body><section><h2>Task overview</h2><table><tr>
    <td>m-nz-cattle</td><td>miou</td><td>4 full</td><td>n/a</td>
    <td>0.8477 exp-cattle by manual</td><td>n/a</td><td>version</td><td>0.8432 test, once</td>
    </tr></table></section><section><h2>m-nz-cattle <span>segmentation</span></h2>
    <h3>Selected checkpoint</h3><p>exp-cattle</p>
    <h3>Fixed validation examples</h3><div class='examples'>saved examples</div>
    <h3>Compute</h3><p>12 min</p></section></body></html>"""
    result = upgrade_document(source)
    assert '84.8%' in result and 'Cattle mapping' in result
    assert 'data-task-button="m-nz-cattle"' in result
    assert 'data-task-button="m-eurosat"' not in result
    assert 'data-task-panel="m-nz-cattle"' in result
    assert 'saved examples' in result and '0.8432 test, once' in result
    assert '<script src=' not in result
