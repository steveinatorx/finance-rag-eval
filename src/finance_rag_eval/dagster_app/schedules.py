"""Dagster schedules (disabled by default)."""

# Schedules are optional and disabled by default
# Uncomment and configure as needed

# from dagster import ScheduleDefinition
# from finance_rag_eval.dagster_app.jobs import rag_offline_job

# daily_rag_schedule = ScheduleDefinition(
#     job=rag_offline_job,
#     cron_schedule="0 0 * * *",  # Daily at midnight
#     name="daily_rag_schedule",
# )
