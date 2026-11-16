# Starting the scans on time (cron-job.org)

GitHub runs scheduled workflows late, sometimes by hours, and under heavy load it skips some.
A workflow started on request (a "workflow dispatch") usually starts within seconds. So a free outside
timer, [cron-job.org](https://cron-job.org), sends that request at an exact time every weekday.
GitHub still does all the work: the same scan, Telegram message, saved state and dashboard.

GitHub's own timers stay on as a **backup**: India 18:11 and 19:47 IST, US 22:23 UTC. If both
the outside timer and GitHub's timer start a run, the second one sees the day was already sent
and stays silent.

| Job | Runs | Message arrives |
|---|---|---|
| India | Mon–Fri 18:00, Asia/Kolkata | about 18:03 IST |
| US | Mon–Fri 17:00, America/New_York | about 7 minutes later (03:37 IST in winter, 02:37 IST in summer) |

## 1. Create a GitHub token

1. Open <https://github.com/settings/personal-access-tokens/new>
   (GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token).
2. **Token name:** `cron-job.org scan trigger`
3. **Expiration:** 1 year. Write the date down; see "When the token expires" below.
4. **Repository access:** *Only select repositories* → `momentum-scanner`.
5. **Permissions → Repository permissions → Actions:** *Read and write*. Leave everything else
   alone (GitHub adds *Metadata: Read-only* by itself).
6. **Generate token** and copy it (it starts with `github_pat_`). GitHub shows it only once.
   Don't paste it into a chat, a file or the repo; it goes only into cron-job.org.

What the token can do: start, cancel, re-run and delete workflow runs and Actions caches in this
one repository. It can't read or change code, secrets or settings.

## 2. Create the two jobs on cron-job.org

Sign up at <https://cron-job.org> (free), then **Create cronjob** once per market. The form has
three tabs: COMMON, NOTIFICATIONS and ADVANCED.

| Setting | India job | US job |
|---|---|---|
| COMMON → **Title** | `momentum-scanner India` | `momentum-scanner US` |
| COMMON → **URL** (replace the `http://` that's already there) | `https://api.github.com/repos/asees543/momentum-scanner/actions/workflows/scan-india.yml/dispatches` | `https://api.github.com/repos/asees543/momentum-scanner/actions/workflows/scan-us.yml/dispatches` |
| COMMON → **Execution schedule** | **Custom**: weekdays Monday–Friday, hour 18, minute 0 | **Custom**: weekdays Monday–Friday, hour 17, minute 0 |
| COMMON → **Crontab expression** should then read | `0 18 * * 1-5` | `0 17 * * 1-5` |
| ADVANCED → **Time zone** | `Asia/Kolkata` | `America/New_York` |

Everything else is the same for both jobs:

- COMMON: **Enable job** on. **Save responses in job history** on (handy if something fails).
- ADVANCED → **Request method:** `POST`
- ADVANCED → **Headers:** press **ADD** four times:

  | Key | Value |
  |---|---|
  | `Accept` | `application/vnd.github+json` |
  | `Authorization` | `Bearer ` followed by your token, e.g. `Bearer github_pat_...` |
  | `X-GitHub-Api-Version` | `2022-11-28` |
  | `Content-Type` | `application/json` |

- ADVANCED → **Request body:**

  ```json
  {"ref":"main","inputs":{"dry_run":"false"}}
  ```

  `dry_run` must be `"false"`: a manual start defaults to a dry run, which only prints the alerts in
  the log and sends nothing.
- ADVANCED → **Timeout:** leave at 30 seconds (GitHub answers in about one).
- NOTIFICATIONS: turn on **execution of the cronjob fails** (notify after 1 failure) and keep
  **the cronjob will be disabled because of too many failures** on.

Before saving, look at **Next executions** on the COMMON tab. It should say *In this job's individual
timezone (Asia/Kolkata)* (or America/New_York) and list only weekdays at 6:00 PM (or 5:00 PM).

## 3. Test each job

Use the job's test run. A good response is **HTTP 204** (no content). Then check
<https://github.com/asees543/momentum-scanner/actions>: a new run started by `workflow_dispatch`
should appear within seconds.

Testing sends no extra Telegram message if that market's latest session was already sent. The run
sees the same market date and stops.

| Response | Meaning |
|---|---|
| 204 | Started. |
| 401 | Token wrong or expired, or the `Bearer ` prefix is missing. |
| 403 or 404 | Token doesn't have *Actions: Read and write*, wasn't given access to `momentum-scanner`, or the URL has a typo. |
| 422 | The request body is wrong (check the quotes and `"ref":"main"`). |

## When the token expires

cron-job.org emails you when a job fails, and GitHub's backup timers keep the alerts coming (later
than usual). Create a new token as in step 1 and paste it into the `Authorization` header of both
jobs (`Bearer ` + token). cron-job.org disables a job after it fails about 25 times in a row, so
renew within a few weeks of the first failure email, or re-enable the job afterwards.
