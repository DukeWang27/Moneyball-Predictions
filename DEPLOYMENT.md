# v0.13.0 Deployment Checklist

## 1. Install the update locally

```bash
cd ~/Desktop/baseball/Moneyball-Predictions
git add .
git commit -m "Checkpoint before v0.13.0" || true
unzip -o ~/Downloads/moneyball-polymarket-v0.13.0.zip -d .
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

## 2. Create Neon

1. Create a Neon project.
2. Open **Connect**.
3. Copy the pooled connection string into `DATABASE_URL`.
4. Disable pooling/copy the direct connection string into `DATABASE_URL_DIRECT`.
5. Keep both strings private.

Create `.env`:

```bash
cp .env.example .env
```

Edit it:

```bash
nano .env
```

Load the variables and migrate:

```bash
set -a
source .env
set +a
alembic upgrade head
python scripts/init_postgres.py
```

## 3. Import existing bets before deployment

```bash
bash scripts/start_local.sh
```

Open the same origin used previously, normally:

```text
http://127.0.0.1:8000
```

On Overview, click **Import bets**. Verify both ledgers under **Paper Portfolio**.

## 4. Push to GitHub

```bash
git add .
git commit -m "Add v0.13.0 Postgres ledgers prop families and Vercel deployment"
git push origin main
```

## 5. Create the Vercel project

1. Open Vercel.
2. Add New → Project.
3. Import `DukeWang27/Moneyball-Predictions`.
4. Keep the repository root as the Root Directory.
5. Add the environment variables below before deploying.

```text
DATABASE_URL=<Neon pooled URL>
DASHBOARD_USERNAME=moneyball
DASHBOARD_PASSWORD=<strong private password>
CRON_SECRET=<long random secret>
```

Generate secrets locally:

```bash
python - <<'PY'
import secrets
print("DASHBOARD_PASSWORD=" + secrets.token_urlsafe(24))
print("CRON_SECRET=" + secrets.token_urlsafe(32))
PY
```

Deploy, then verify:

```text
https://YOUR-PROJECT.vercel.app/health
```

## 6. Configure external minute scheduler

Create two GET jobs that run every minute:

```text
https://YOUR-PROJECT.vercel.app/api/internal/cron/lineups
https://YOUR-PROJECT.vercel.app/api/internal/cron/settle-paper-bets
```

Add this request header to each:

```text
Authorization: Bearer YOUR_CRON_SECRET
```

The endpoint performs cadence skipping, so a one-minute trigger does not fetch every game every minute.

## 7. Production checks

- Overview displays two $100 ledgers.
- Imported historical bets appear under the correct ledger.
- Moneyline bets only appear after both lineups are confirmed.
- Each pitcher has one grouped family card.
- Only one threshold/side per pitcher shows `BEST BET`.
- An inverted market ladder shows a warning and no bet button.
- `/health` reports Postgres as healthy.
- Cron requests return HTTP 200 with count summaries.
