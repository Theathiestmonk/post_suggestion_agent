# job_queue.py - Simple job system for RL learning

import asyncio
import time
import threading
from queue import Queue
from typing import Dict, Any, Optional
from datetime import datetime
import pytz

# Indian Standard Time (IST) - Asia/Kolkata
IST = pytz.timezone("Asia/Kolkata")
import db
import rl_agent
import snaphot_collector
# Note: check_and_run_scheduled_jobs is imported dynamically to avoid circular imports

# Thread-safe job queue (replace with Redis/Celery for production)
job_queue = Queue()
job_results = {}  # Store job results by job_id
running_jobs = set()  # Track running job IDs

class Job:
    def __init__(self, job_type: str, job_id: str, payload: Dict[str, Any]):
        self.job_type = job_type  # "reward_calculation" or "rl_update"
        self.job_id = job_id
        self.payload = payload
        self.created_at = datetime.now(IST)
        self.status = "queued"

async def process_reward_calculation_job(job: Job) -> Dict[str, Any]:
    """Process reward calculation job"""
    try:
        payload = job.payload
        profile_id = payload["profile_id"]
        post_id = payload["post_id"]
        platform = payload["platform"]

        print(f"Processing reward calculation for {post_id} on {platform}")

        # Calculate reward
        result = db.fetch_or_calculate_reward(profile_id, post_id, platform)

        # Debug: Print result
        print(f"🔍 Reward calculation result: {result}")

        if result["status"] == "calculated":
            # Queue RL update job
            rl_job = Job(
                job_type="rl_update",
                job_id=f"rl_{post_id}_{int(time.time())}",
                payload={
                    "profile_id": profile_id,
                    "post_id": post_id,
                    "platform": platform,
                    "reward_value": result["reward"]
                }
            )
            job_queue.put(rl_job)
            print(f"📋 Queued RL update job for {post_id}")

        return result

    except Exception as e:
        print(f"❌ Error in reward calculation job: {e}")
        return {"status": "error", "error": str(e)}

async def process_rl_update_job(job: Job) -> Dict[str, Any]:
    """Process RL update job"""
    try:
        payload = job.payload
        profile_id = payload["profile_id"]
        post_id = payload["post_id"]
        platform = payload["platform"]
        reward_value = payload["reward_value"]

        print(f"🧠 Processing RL update for {post_id} (reward: {reward_value:.4f})")

        # Get action and context from database
        # This assumes the action and context are stored during posting
        action_data = get_action_and_context_from_db(post_id, platform, profile_id)

        if not action_data:
            print(f"⚠️  No action data found for {post_id}, skipping RL update")
            return {"status": "skipped", "reason": "no_action_data"}

        action = action_data["action"]
        context = action_data["context"]
        ctx_vec = action_data["ctx_vec"]

        # Get current baseline using pure mathematical update
        current_baseline = db.update_baseline_mathematical(platform, reward_value, beta=0.1)

        # Update RL
        rl_agent.update_rl(
            context=context,
            action=action,
            ctx_vec=ctx_vec,
            reward=reward_value,
            baseline=current_baseline
        )

        print(f"✅ RL update completed for {post_id}")
        return {"status": "completed", "baseline": current_baseline}

    except Exception as e:
        print(f"❌ Error in RL update job: {e}")
        return {"status": "error", "error": str(e)}

def get_action_and_context_from_db(post_id: str, platform: str, profile_id: str) -> Optional[Dict[str, Any]]:
    """Get action and context data from database for RL update"""
    try:
        # Get action data from rl_actions table
        action_result = db.supabase.table("rl_actions").select("*").eq("post_id", post_id).eq("platform", platform).execute()

        if not action_result.data:
            return None

        action_row = action_result.data[0]

        # Reconstruct action dict
        action = {
            "HOOK_TYPE": action_row.get("hook_type"),
            "INFORMATION_DEPTH": action_row.get("information_depth"),
            "TONE": action_row.get("tone"),
            "CREATIVITY": action_row.get("creativity"),
            "COMPOSITION_STYLE": action_row.get("composition_style"),
            "VISUAL_STYLE": action_row.get("visual_style")
        }

        # Get the topic from the post data
        topic = action_row.get("topic", "")
        topic_embedding = db.embed_topic(topic) if topic else None

        # Get business embedding
        business_embedding = db.get_profile_embedding_with_fallback(profile_id)
        if business_embedding is None:
            print(f"❌ No business embedding found for {profile_id}, cannot perform RL update")
            return None

        # Use topic embedding if available, otherwise use business embedding
        final_topic_embedding = topic_embedding if topic_embedding is not None else business_embedding

        # Reconstruct context with real business data
        context = {
            "platform": platform,
            "time_bucket": action_row.get("time_bucket"),
            "business_embedding": business_embedding,
            "topic_embedding": final_topic_embedding
        }

        # Reconstruct context vector
        from rl_agent import build_context_vector
        ctx_vec = build_context_vector(context)

        return {
            "action": action,
            "context": context,
            "ctx_vec": ctx_vec
        }

    except Exception as e:
        print(f"❌ Error retrieving action data for {post_id}: {e}")
        return None

def job_worker():
    """Main job processing worker (synchronous)"""
    print("🚀 Starting RL job worker...")

    while True:
        try:
            print("Job worker waiting for jobs...")
            job = job_queue.get()  # Blocking get
            print(f"📥 Job worker received job: {job.job_id}")
            job.status = "running"
            running_jobs.add(job.job_id)

            print(f"📋 Processing job {job.job_id} ({job.job_type})")

            # Create new event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                if job.job_type == "reward_calculation":
                    result = loop.run_until_complete(process_reward_calculation_job(job))
                elif job.job_type == "rl_update":
                    result = loop.run_until_complete(process_rl_update_job(job))
                else:
                    result = {"status": "error", "error": f"Unknown job type: {job.job_type}"}

                # Debug: Print job completion
                print(f"✅ Job {job.job_id} completed with result: {result}")

                job_results[job.job_id] = result
            finally:
                loop.close()

            running_jobs.remove(job.job_id)

        except Exception as e:
            print(f"❌ Job worker error: {e}")
            time.sleep(1)  # Brief pause on error

def queue_reward_calculation_job(profile_id: str, post_id: str, platform: str) -> str:
    """Queue a reward calculation job"""
    job_id = f"reward_{post_id}_{int(time.time())}"
    job = Job(
        job_type="reward_calculation",
        job_id=job_id,
        payload={
            "profile_id": profile_id,
            "post_id": post_id,
            "platform": platform
        }
    )

    job_queue.put(job)
    print(f"📋 Queued reward calculation job: {job_id}")
    print(f"📊 Current queue size: {job_queue.qsize()}")
    return job_id

async def run_job_worker_async():
    """Async wrapper for the synchronous job worker"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, job_worker)


def check_and_retry_missed_posts():
    """Check for missed posts and retry failed ones"""
    print("🔄 Starting post rechecking process...")

    try:
        # Get all active business profiles
        all_business_ids = db.get_all_profile_ids()
        print(f"📊 Found {len(all_business_ids)} business profiles to check")

        rechecked_count = 0

        for business_id in all_business_ids:
            try:
                print(f"\n🏢 Rechecking business: {business_id}")

                # Get connected platforms for this business
                user_connected_platforms = list(set(db.get_connected_platforms(business_id)))
                print(f"📱 Business {business_id} has {len(user_connected_platforms)} connected platforms: {user_connected_platforms}")

                for platform in user_connected_platforms:
                    try:
                        platform = platform.lower().strip()  # normalize

                        if platform not in ["instagram", "facebook"]:
                            print(f"❌ Skipping unsupported platform: {platform} for business {business_id}")
                            continue

                        # Check for posts that should have been created but weren't
                        missed_posts = check_missed_posts(business_id, platform)
                        if missed_posts:
                            print(f"⚠️ Found {len(missed_posts)} missed posts for {business_id} on {platform}")
                            for post_info in missed_posts:
                                retry_missed_post(business_id, platform, post_info)
                                rechecked_count += 1

                        # Check for failed posts that need retrying
                        failed_posts = check_failed_posts(business_id, platform)
                        if failed_posts:
                            print(f"⚠️ Found {len(failed_posts)} failed posts for {business_id} on {platform}")
                            for post_info in failed_posts:
                                retry_failed_post(business_id, platform, post_info)
                                rechecked_count += 1

                    except Exception as e:
                        print(f"❌ Failed to recheck posts for {business_id} on {platform}: {e}")
                        continue

            except Exception as e:
                print(f"❌ Failed to recheck business {business_id}: {e}")
                continue

        print(f"✅ Post rechecking completed. Processed {rechecked_count} posts")

    except Exception as e:
        print(f"❌ Critical error in post rechecking: {e}")


def check_missed_posts(business_id: str, platform: str) -> list:
    """Check for posts that should have been created but weren't"""
    try:
        # Get today's date
        today = datetime.now(IST).date()

        # Check if posts exist for today
        res = db.supabase.table("post_contents").select("post_id").eq("business_id", business_id).eq("platform", platform).eq("post_date", today.isoformat()).execute()

        # If no posts exist for today and business should post today, mark as missed
        if not res.data and db.should_create_post_today(business_id):
            return [{"date": today.isoformat(), "reason": "missing_daily_post"}]
        else:
            return []

    except Exception as e:
        print(f"Error checking missed posts for {business_id} on {platform}: {e}")
        return []


def check_failed_posts(business_id: str, platform: str) -> list:
    """Check for posts that failed during creation or posting"""
    try:
        # Look for posts with error status or stuck in generated state too long
        # Query for posts with error or generated status
        error_res = (db.supabase.table("post_contents")
                     .select("post_id, status, created_at")
                     .eq("business_id", business_id)
                     .eq("platform", platform)
                     .eq("status", "error")
                     .execute())

        generated_res = (db.supabase.table("post_contents")
                         .select("post_id, status, created_at")
                         .eq("business_id", business_id)
                         .eq("platform", platform)
                         .eq("status", "generated")
                         .execute())

        # Combine results
        all_posts = []
        if error_res.data:
            all_posts.extend(error_res.data)
        if generated_res.data:
            all_posts.extend(generated_res.data)

        res = type('Result', (), {'data': all_posts})()

        failed_posts = []
        current_time = datetime.now(IST)

        for post in res.data or []:
            created_at = datetime.fromisoformat(post["created_at"].replace('Z', '+00:00'))
            if created_at.tzinfo is None:
                created_at = pytz.utc.localize(created_at)
            created_at_ist = created_at.astimezone(IST)

            # If post has been in error/generated state for more than 2 hours, consider it failed
            if (current_time - created_at_ist).total_seconds() > 7200:  # 2 hours
                failed_posts.append({
                    "post_id": post["post_id"],
                    "status": post["status"],
                    "reason": "stuck_too_long"
                })

        return failed_posts

    except Exception as e:
        print(f"Error checking failed posts for {business_id} on {platform}: {e}")
        return []


def retry_missed_post(business_id: str, platform: str, post_info: dict):
    """Retry creating a missed post"""
    try:
        print(f"🔄 Retrying missed post for {business_id} on {platform} (date: {post_info['date']})")

        # Import main module functions dynamically
        import main

        # Create the missed post
        main.run_one_post(
            BUSINESS_ID=business_id,
            platform=platform,
        )

        print(f"✅ Successfully retried missed post for {business_id} on {platform}")

    except Exception as e:
        print(f"❌ Failed to retry missed post for {business_id} on {platform}: {e}")


def retry_failed_post(business_id: str, platform: str, post_info: dict):
    """Retry a failed post"""
    try:
        print(f"🔄 Retrying failed post {post_info['post_id']} for {business_id} on {platform}")

        # For now, just mark as needing attention - could implement more sophisticated retry logic
        # This could involve re-generating content or re-attempting posting

        # Option 1: Delete and recreate the post
        # Option 2: Just update status to trigger re-processing
        # For simplicity, let's delete and recreate

        post_id = post_info["post_id"]

        # Delete the failed post (this will cascade to related records)
        db.supabase.table("post_contents").delete().eq("post_id", post_id).execute()

        # Create a new post to replace it
        import main
        main.run_one_post(
            BUSINESS_ID=business_id,
            platform=platform,
        )

        print(f"✅ Successfully retried failed post {post_id} for {business_id} on {platform}")

    except Exception as e:
        print(f"❌ Failed to retry failed post {post_info['post_id']} for {business_id} on {platform}: {e}")


async def run_cron_job_services(duration_seconds: int = 90):
    """Run job processing and metrics collection for a limited time (suitable for cron jobs)"""
    print(f"🚀 Starting cron job services for {duration_seconds} seconds...")

    # Create tasks for both services
    job_task = asyncio.create_task(run_job_worker_async())
    metrics_task = asyncio.create_task(snaphot_collector.run_continuous_metrics_collection())

    # Wait for the specified duration, then cancel tasks
    try:
        await asyncio.wait_for(asyncio.gather(job_task, metrics_task, return_exceptions=True), timeout=duration_seconds)
    except asyncio.TimeoutError:
        print(f"⏰ Cron job duration ({duration_seconds}s) reached, stopping services...")
        # Cancel the tasks
        job_task.cancel()
        metrics_task.cancel()

        # Wait for cancellation to complete
        try:
            await job_task
        except asyncio.CancelledError:
            pass
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass

    print("✅ Cron job services completed")


if __name__ == "__main__":
    """Run cron job services (rechecking + limited job processing)"""
    print("🔄 Starting Cron Job RL Service...")
    print("📋 This will run rechecking and process jobs for a limited time")
    print("⚠️  Designed for cron job execution every 15 minutes")

    try:
        # Run rechecking to catch missed/failed posts
        print("🔄 Running post rechecking...")
        check_and_retry_missed_posts()

        # Run services for a limited time (90 seconds to fit within 15-minute cron cycles)
        asyncio.run(run_cron_job_services(duration_seconds=90))

        print("✅ Cron job cycle completed successfully")

    except Exception as e:
        print(f"❌ Cron job error: {e}")
        raise






