"""Job feeder for the 100-job submit cap (QOSMaxSubmitJobPerUser): holds jobs in a queue file and submits them in order
while fewer than CAP of the user's jobs are queued; dependencies by key (afterok); failed jobs retried twice on
another node (as retry_watchdog). Queue file: logs/feeder_queue.json = list of {key, sbatch, dep, id, retries}.
Add jobs with `feeder.py add KEY "SBATCH ARGS" [DEP_KEY]` (safe while the feeder runs: guarded by a lock file)."""
import fcntl, json, subprocess, sys, time
from pathlib import Path
REPO = Path("/home/jeffc2/ANLP_final/xling-pilots"); Q = REPO / "logs" / "feeder_queue.json"; LOCK = REPO / "logs" / ".feeder.lock"
CAP = 92
BAD = {"babel-u9-20", "babel-w9-28", "babel-w9-26", "babel-l9-32"}
FAIL = {"FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "TIMEOUT", "BOOT_FAIL", "DEADLINE"}
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True, cwd=REPO).stdout.strip()
def log(m): print(time.strftime("%F %T"), m, flush=True)
class locked:
    def __enter__(self):
        self.f = open(LOCK, "w"); fcntl.flock(self.f, fcntl.LOCK_EX)
        self.q = json.loads(Q.read_text()) if Q.exists() else []; return self.q
    def __exit__(self, *a):
        Q.write_text(json.dumps(self.q, indent=1)); fcntl.flock(self.f, fcntl.LOCK_UN)
def submit(j, dep_id):
    d = f"--dependency=afterok:{dep_id}" if dep_id else ""
    return sh(f"sbatch --parsable --exclude={','.join(sorted(BAD))} {d} {j['sbatch']}")
def step():
    with locked() as q:
        by = {j["key"]: j for j in q}
        ids = [j["id"] for j in q if j["id"]]
        acct = {}
        if ids:
            for line in sh(f"sacct -X -n -P -j {','.join(ids)} -o JobID,State,NodeList").splitlines():
                jid, st, node = line.split("|"); acct[jid] = (st.split()[0], node)
        n_queued = len(sh("squeue -h -u $USER").splitlines())
        for j in q:
            dep = by.get(j.get("dep")) if j.get("dep") else None
            st, node = acct.get(j["id"], ("?", "")) if j["id"] else ("UNSUBMITTED", "")
            dep_st = acct.get(dep["id"], ("?",))[0] if dep and dep["id"] else None
            if st == "UNSUBMITTED" or (st in FAIL and j["retries"] < 2) or st == "CANCELLED" and dep_st in FAIL and dep and dep["retries"] <= 2 and j.get("rechain", 0) < 3:
                if dep and not dep["id"]:
                    continue          # dependency not submitted yet
                if n_queued >= CAP:
                    break
                if st in FAIL:
                    BAD.add(node); j["retries"] += 1
                if st == "CANCELLED":
                    j["rechain"] = j.get("rechain", 0) + 1
                live_dep = dep["id"] if dep and dep_st not in ("COMPLETED",) else None
                j["id"] = submit(j, live_dep); n_queued += 1
                log(f"{j['key']}: {st} -> submitted {j['id']}" + (f" after {live_dep}" if live_dep else ""))
        return all(j["id"] and acct.get(j["id"], ("?",))[0] in ({"COMPLETED"} | FAIL) and (acct[j["id"]][0] == "COMPLETED" or j["retries"] >= 2) for j in q)
if __name__ == "__main__":
    if sys.argv[1:2] == ["add"]:
        key, sb = sys.argv[2], sys.argv[3]; dep = sys.argv[4] if len(sys.argv) > 4 else None
        with locked() as q:
            if any(j["key"] == key for j in q): print("exists", key)
            else: q.append({"key": key, "sbatch": sb, "dep": dep, "id": None, "retries": 0}); print("queued", key)
    else:
        while True:
            try: step()
            except Exception as e: log(f"error {e!r}")
            time.sleep(300)
