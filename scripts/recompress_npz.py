import argparse
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _needs_recompress(path: str) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            infos = zf.infolist()
            if not infos:
                return False
            return any(i.compress_type == zipfile.ZIP_STORED for i in infos)
    except Exception:
        return False


def _recompress_one(args):
    path, force, dry_run = args
    path = str(path)
    try:
        old_size = os.path.getsize(path)
        if not force and not _needs_recompress(path):
            return ("skipped", path, old_size, old_size, None)
        if dry_run:
            return ("would_recompress", path, old_size, old_size, None)

        tmp_path = str(Path(path).with_suffix(".tmp.npz"))
        try:
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            np.savez_compressed(tmp_path, **arrays)
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        new_size = os.path.getsize(path)
        return ("recompressed", path, old_size, new_size, None)
    except Exception as e:
        return ("failed", path, None, None, str(e))


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompress existing .npz files in-place (np.savez -> np.savez_compressed).")
    ap.add_argument("--data-dir", default="federated_data", help="Directory containing *.npz files.")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 4))
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of files processed.")
    ap.add_argument("--force", action="store_true", help="Recompress even if file already looks compressed.")
    ap.add_argument("--dry-run", action="store_true", help="Only report what would change.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"--data-dir not found: {data_dir}")

    files = sorted(str(p) for p in data_dir.rglob("*.npz"))
    if args.limit is not None:
        files = files[: int(args.limit)]
    if not files:
        raise SystemExit(f"No .npz files found under: {data_dir}")

    workers = max(1, int(args.workers))
    tasks = [(p, bool(args.force), bool(args.dry_run)) for p in files]

    stats = {"recompressed": 0, "skipped": 0, "would_recompress": 0, "failed": 0}
    old_total = 0
    new_total = 0
    failures = []

    iterator = None
    if tqdm is not None:
        iterator = tqdm(total=len(tasks), desc="recompress", unit="file")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_recompress_one, t) for t in tasks]
        for fut in as_completed(futures):
            status, path, old_size, new_size, err = fut.result()
            stats[status] = stats.get(status, 0) + 1
            if isinstance(old_size, int):
                old_total += old_size
            if isinstance(new_size, int):
                new_total += new_size
            if status == "failed":
                failures.append({"file": path, "error": err})
            if iterator is not None:
                iterator.update(1)

    if iterator is not None:
        iterator.close()

    print("Summary:", stats)
    if old_total and (stats.get("recompressed", 0) > 0) and not bool(args.dry_run):
        saved = old_total - new_total
        pct = 100.0 * (float(saved) / float(old_total)) if old_total else 0.0
        print(f"Total size: {old_total/1024/1024/1024:.2f} GB -> {new_total/1024/1024/1024:.2f} GB (saved {saved/1024/1024/1024:.2f} GB, {pct:.1f}%)")
    if failures:
        print(f"Failures ({len(failures)}):")
        for f in failures[:20]:
            print(f"- {f['file']}: {f['error']}")
        if len(failures) > 20:
            print(f"... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()

