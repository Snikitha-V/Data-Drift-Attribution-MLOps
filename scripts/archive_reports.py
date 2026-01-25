import zipfile
from pathlib import Path
from datetime import datetime

REPORTS = Path('reports')
ARCHIVE_DIR = REPORTS / 'archive'
ARCHIVE_DIR.mkdir(exist_ok=True)

# patterns to archive
patterns = [
    'drift_window_*.csv',
    'attribution_window_*.csv',
    'attribution_window_*.png'
]

# files to keep
keep_files = {
    'drift_summary.csv',
    'attribution_all_windows.csv',
    'shap_importance.csv',
    'attribution_summary.csv',
    'attribution_timeline.png',
    'window_sizes.png'
}

now = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
archive_name = ARCHIVE_DIR / f'paysim_windows_{now}.zip'

files_to_archive = []
for p in patterns:
    for f in REPORTS.glob(p):
        if f.name not in keep_files:
            files_to_archive.append(f)

if not files_to_archive:
    print('No per-window files found to archive.')
else:
    with zipfile.ZipFile(archive_name, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in files_to_archive:
            z.write(f, arcname=f.name)
    # remove originals
    for f in files_to_archive:
        f.unlink()

    print(f'Archived {len(files_to_archive)} files to {archive_name}')