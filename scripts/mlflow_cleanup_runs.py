import os
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
import dagshub


def _to_epoch_ms(iso: str) -> int:
    dt = datetime.fromisoformat(iso.replace('Z', '+00:00'))
    return int(dt.timestamp() * 1000)


def _in_range(ts_ms: int, start_ms: int, end_ms: int) -> bool:
    if ts_ms is None:
        return False
    return start_ms <= ts_ms <= end_ms


def main():
    owner = os.environ.get('DAGSHUB_OWNER')
    repo = os.environ.get('DAGSHUB_REPO')
    token = os.environ.get('DAGSHUB_TOKEN')
    tracking_uri_env = os.environ.get('MLFLOW_TRACKING_URI')
    if owner and repo:
        dagshub.init(repo, owner, mlflow=True)
    if tracking_uri_env:
        mlflow.set_tracking_uri(tracking_uri_env)

    start_iso = os.environ.get('PERIOD_START_ISO', '2022-01-01T00:00:00Z')
    end_iso = os.environ.get('PERIOD_END_ISO', '2022-06-30T23:59:59Z')
    start_ms = _to_epoch_ms(start_iso)
    end_ms = _to_epoch_ms(end_iso)

    exp_filter = os.environ.get('MLFLOW_EXPERIMENT_FILTER')  # regex or name match (simple contains)
    client = MlflowClient()
    try:
        experiments = client.search_experiments()
    except Exception:
        experiments = []
    target_exps = []
    for e in experiments:
        if not exp_filter or (exp_filter in (e.name or '')):
            target_exps.append(e)

    deleted = 0
    kept = 0
    for exp in target_exps:
        runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=10000)
        for r in runs:
            st = r.info.start_time
            et = r.info.end_time
            in_period = _in_range(st, start_ms, end_ms) and (et is None or _in_range(et, start_ms, end_ms))
            if in_period:
                kept += 1
                continue
            try:
                client.delete_run(r.info.run_id)
                deleted += 1
            except Exception:
                pass

    print('[MLflow Cleanup] Kept:', kept, '| Deleted:', deleted)


if __name__ == '__main__':
    main()