from mlflow.tracking import MlflowClient
client = MlflowClient()
runs = client.search_runs(experiment_ids=None, filter_string=None, run_view_type=1, max_results=50)
print('Found', len(runs), 'runs')
for r in runs[:20]:
    rid = r.info.run_id
    print('RUN', rid, 'status', r.info.status)
    try:
        arts = client.list_artifacts(rid, path='')
        print(' artifacts:', [a.path for a in arts])
    except Exception as e:
        print(' artifacts error', e)
