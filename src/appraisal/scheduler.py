from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger


def build_weekly_retrain_scheduler(cron_expr: str, job_fn) -> BackgroundScheduler:
    fields = cron_expr.split()
    if len(fields) != 5:
        raise ValueError("Cron must have 5 fields: min hour day month day_of_week")
    minute, hour, day, month, day_of_week = fields
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        job_fn,
        trigger=CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
        ),
        id="weekly_retrain",
        replace_existing=True,
    )
    return scheduler

