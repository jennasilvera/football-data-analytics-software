from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from football_analytics.api.main import create_app
from football_analytics.config import Settings
from football_analytics.storage.repository import Repository, _insert_records, _prepare_records

START = datetime(2020, 1, 1, tzinfo=UTC)
CUTOFF = START + timedelta(days=2)


@pytest.mark.parametrize('path,kind', [
    ('teams', 'team'), ('matches', 'match'), ('predictions', 'forecast'),
    ('forecast-batches', 'forecast_batch'), ('models', 'model'),
])
def test_pages_exclude_later_commits_and_revisions(tmp_path, path, kind):
    app = create_app(Settings(database=tmp_path / 'db', allow_unauthenticated=True))
    repo = app.state.repository
    def payload(value):
        return {'value': value, 'training_cutoff': value, 'transforms': {}}
    repo.put_many([(kind, key, payload(key), START) for key in ['a', 'c']], recorded_at=START)
    client = TestClient(app)
    # The writer has already inserted historical records, but has not committed.
    with repo.connection() as writer:
        writer.execute('BEGIN IMMEDIATE')
        _insert_records(writer, _prepare_records([
            (kind, 'b', payload('b'), START), (kind, 'c', payload('changed'), START),
        ], START + timedelta(days=1)))
        first = client.get('/' + path, params={'limit': 1, 'as_of': CUTOFF.isoformat()}).json()
    params = {'limit': 10, 'after': first['next_cursor'], 'as_of': first['as_of']}
    params['through_sequence'] = first['through_sequence']
    second = client.get('/' + path, params=params).json()
    field = 'training_cutoff' if kind == 'model' else 'value'
    assert [r[field] for r in second['items']] == ['c']
    assert second['through_sequence'] == first['through_sequence']
    fresh = client.get('/' + path, params={'as_of': CUTOFF.isoformat()}).json()
    assert [r[field] for r in fresh['items']] == ['a', 'b', 'changed']
    assert client.get(
        '/' + path, params={'through_sequence': first['through_sequence']}
    ).status_code == 422
    for value in [-1, 2**63, fresh["through_sequence"] + 1]:
        assert client.get(
            '/' + path, params={'through_sequence': value, 'as_of': first['as_of']}
        ).status_code == 422


def test_scan_keeps_one_committed_snapshot(tmp_path, monkeypatch):
    repo = Repository(tmp_path / 'db')
    repo.put_many(
        [('team', f'{i:04}', {'value': i}, START) for i in range(1001)], recorded_at=START
    )
    original = repo.list
    calls = 0
    def concurrent_list(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            repo.put_many([
                ('team', '1000', {'value': -1}, START),
                ('team', '1001', {'value': 1001}, START),
            ], recorded_at=START + timedelta(days=1))
        return original(*args, **kwargs)
    monkeypatch.setattr(repo, 'list', concurrent_list)
    rows = repo.scan('team', as_of=CUTOFF)
    assert len(rows) == 1001
    assert rows[-1]['payload']['value'] == 1000


def test_empty_snapshot_stays_empty(tmp_path):
    repo = Repository(tmp_path / 'db')
    boundary = repo.snapshot_sequence()
    assert boundary == 0
    repo.put('team', 'a', {}, available_at=START, recorded_at=START)
    assert repo.list('team', as_of=CUTOFF, through_sequence=boundary) == []
    for value in [-1, 2**63, True, 1.5]:
        with pytest.raises(ValueError):
            repo.list('team', through_sequence=value)
