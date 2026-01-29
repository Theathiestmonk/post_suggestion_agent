# db.py
import os
import math
import uuid
import numpy as np
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime, timedelta, timezone
import pytz
from typing import List

# Load environment variables from .env file
load_dotenv()

# Reward weights for different time periods (higher weight = more important)
REWARD_WEIGHTS = {     # at alpha = 0.35
    6:   0.396,
    24:  0.258,
    48:  0.168,
    72:  0.109,
    168: 0.071
}

# Indian Standard Time (IST) - Asia/Kolkata
IST = pytz.timezone("Asia/Kolkata")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing required environment variables: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY")

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    raise ValueError(f"Failed to create Supabase client: {e}")

def calculate_platform_engagement(platform: str, metrics: dict) -> float:
    """
    Calculate platform-specific engagement score.
    Shared utility function used by both reward calculation methods.
    """
    if platform == "instagram":
        # Instagram values SAVES the most
        return (
            3.0 * metrics.get("saves", 0) +
            2.0 * metrics.get("shares", 0) +
            1.0 * metrics.get("comments", 0) +
            0.3 * metrics.get("likes", 0)
        )
    elif platform == "x":
        # X values REPLIES the most
        return (
            3.0 * metrics.get("replies", 0) +
            2.0 * metrics.get("retweets", 0) +
            1.0 * metrics.get("likes", 0)
        )
    elif platform == "linkedin":
        # LinkedIn values COMMENTS + SHARES
        return (
            3.0 * metrics.get("comments", 0) +
            2.0 * metrics.get("shares", 0) +
            1.0 * metrics.get("likes", 0)
        )
    elif platform == "facebook":
        # Facebook values COMMENTS + SHARES
        return (
            3.0 * metrics.get("comments", 0) +
            2.0 * metrics.get("shares", 0) +
            1.0 * metrics.get("reactions", 0)
        )
    else:
        raise ValueError(f"Unsupported platform: {platform}")

# ---------- PREFERENCES ----------


def get_preference(platform, time_bucket, dimension, value):
    try:
        res = supabase.table("rl_preferences") \
            .select("preference_score") \
            .eq("platform", platform) \
            .eq("time_bucket", time_bucket) \
            .eq("dimension", dimension) \
            .eq("action_value", value) \
            .execute()

        if res.data and len(res.data) > 0 and "preference_score" in res.data[0]:
            return float(res.data[0]["preference_score"])
        return 0.0
    except Exception as e:
        print(f"Error getting preference for {platform}, {dimension}={value}: {e}")
        return 0.0


def get_preferences_batch(platform, time_bucket) -> dict:
    """
    Fetch ALL preferences for a context in one go.
    Returns: dict { (dimension, value): score }
    """
    try:
        res = supabase.table("rl_preferences") \
            .select("dimension, action_value, preference_score") \
            .eq("platform", platform) \
            .eq("time_bucket", time_bucket) \
            .execute()

        prefs = {}
        if res.data:
            for row in res.data:
                key = (row["dimension"], row["action_value"])
                prefs[key] = float(row["preference_score"])
        return prefs

    except Exception as e:
        print(f"Error batch fetching preferences for {platform}: {e}")
        return {}


def update_preference(platform, time_bucket, dimension, value, delta):
    """
    Update preference scores with increment operations.

    NOTE: This implementation still has a potential race condition between
    SELECT and UPDATE operations. For production systems, consider:

    1. Using database triggers for atomic increments
    2. Creating a stored procedure for this operation
    3. Using PostgreSQL's ON CONFLICT DO UPDATE with increment logic
    4. Implementing optimistic locking with version columns

    Current implementation includes error handling and retry logic as mitigation.
    """
    print(f"Updating preference: {platform} | {time_bucket} | {dimension}={value} | delta={delta:.6f}")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = supabase.table("rl_preferences") \
                .select("id, preference_score, num_samples") \
                .eq("platform", platform) \
                .eq("time_bucket", time_bucket) \
                .eq("dimension", dimension) \
                .eq("action_value", value) \
                .execute()

            if res.data and len(res.data) > 0:
                row = res.data[0]
                if "id" in row and "preference_score" in row and "num_samples" in row:
                    current_score = float(row["preference_score"])
                    current_samples = int(row["num_samples"])
                    new_score = current_score + delta

                    print(f"   Existing preference: {current_score:.4f} -> {new_score:.4f} (samples: {current_samples} -> {current_samples + 1})")
                    supabase.table("rl_preferences").update({
                        "preference_score": new_score,
                        "num_samples": current_samples + 1,
                        "updated_at": datetime.now(IST).isoformat()
                    }).eq("id", row["id"]).execute()
                    return  # Success
            else:
                # Insert new preference - use upsert for safety
                print(f"   Creating new preference entry with score: {delta:.4f}")
                try:
                    supabase.table("rl_preferences").upsert({
                        "platform": platform,
                        "time_bucket": time_bucket,
                        "dimension": dimension,
                        "action_value": value,
                        "preference_score": delta,
                        "num_samples": 1
                    }).execute()
                    return  # Success
                except Exception as insert_error:
                    # If upsert fails, the row might have been inserted by another process
                    # Try update instead
                    if attempt < max_retries - 1:  # Don't retry on last attempt
                        continue
                    raise insert_error

            # If we get here, something unexpected happened
            if attempt == max_retries - 1:
                raise ValueError("Failed to update preference after all retries")

        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Error updating preference for {platform}, {dimension}={value} after {max_retries} attempts: {e}")
                raise
            else:
                print(f"Attempt {attempt + 1} failed, retrying: {e}")
                continue
def insert_post_content(
    post_id,
    action_id,
    platform,
    business_id,
    topic,
    image_prompt,
    caption_prompt,
    generated_caption,
    generated_image_url
):
    try:
        # Get scheduling preferences for the business
        scheduling_prefs = get_profile_scheduling_prefs(business_id)
        time_bucket = scheduling_prefs.get("time_bucket", "evening")

        # Map time_bucket to IST times
        time_mapping = {
            "morning": "08:30:00",
            "afternoon": "13:00:00",
            "evening": "18:30:00",
            "night": "21:30:00"
        }

        # Get today's date and scheduled time
        post_date = datetime.now(IST).date().isoformat()
        post_time = time_mapping.get(time_bucket, "18:30:00")  # Default to evening

        supabase.table("post_contents").insert({
            "post_id": post_id,
            "action_id": action_id,
            "platform": platform,
            "business_id": business_id,
            "topic": topic,
            "image_prompt": image_prompt,
            "caption_prompt": caption_prompt,
            "generated_caption": generated_caption,
            "generated_image_url": generated_image_url,
            "post_date": post_date,
            "post_time": post_time

        }).execute()
    except Exception as e:
        print(f"Error inserting post content for post_id {post_id}: {e}")
        raise
        
def mark_post_as_posted(post_id, media_id=None):
    """
    Mark a post as posted with optional media_id
    """
    try:
        update_data = {
            "status": "posted"
        }

        if media_id:
            update_data["media_id"] = media_id

        supabase.table("post_contents").update(update_data).eq("post_id", post_id).execute()
        print(f"✅ Marked post {post_id} as posted")
    except Exception as e:
        print(f"❌ Error marking post {post_id} as posted: {e}")
        raise


# --------------------------------------------------
# Recent topics
# --------------------------------------------------    
def recent_topics(business_id: str, platform: str, limit: int = 10) -> List[str]:
    """
    Fetch the most recent topics from post_contents table for a specific business and platform.
    
    Args:
        business_id: The business/profile ID to get topics for
        platform: The platform to filter topics by
        limit: Number of recent topics to fetch (default: 10)
    
    Returns:
        List of recent topic strings, ordered by most recent first
    """
    try:
        res = supabase.table("post_contents").select("topic").eq("business_id", business_id).eq("platform", platform).order("created_at", desc=True).limit(limit).execute()
        if res.data:
            # Extract topics and filter out None/empty values
            topics = [row["topic"] for row in res.data if row.get("topic")]
            return topics
        else:
            return []
            
    except Exception as e:
        print(f"Error fetching recent topics for business {business_id}: {e}")
        return []


def schedule_post(post_id):
    """
    Mark a post as scheduled
    """
    try:
        supabase.table("post_contents").update({
            "status": "scheduled"
        }).eq("post_id", post_id).execute()
        print(f"📅 Marked post {post_id} as scheduled")
    except Exception as e:
        print(f"❌ Error marking post {post_id} as scheduled: {e}")
        raise

def fail_post(post_id):
    """
    Mark a post as failed
    """
    try:
        supabase.table("post_contents").update({
            "status": "failed"
        }).eq("post_id", post_id).execute()
        print(f"❌ Marked post {post_id} as failed")
    except Exception as e:
        print(f"❌ Error marking post {post_id} as failed: {e}")
        raise

def get_posts_by_status(status):
    """
    Get all posts with a specific status
    """
    try:
        res = supabase.table("post_contents").select("*").eq("status", status).execute()
        return res.data or []
    except Exception as e:
        print(f"❌ Error fetching posts with status '{status}': {e}")
        return []

def get_scheduled_posts_ready_to_post():
    """
    Get scheduled posts that are ready to be posted (current time >= scheduled time)
    """
    try:
        current_time = datetime.now(IST)
        current_date = current_time.date()
        current_time_str = current_time.strftime("%H:%M:%S")

        # Query for posts scheduled for today at or before current time
        res = supabase.table("post_contents") \
            .select("*") \
            .eq("status", "scheduled") \
            .eq("post_date", current_date.isoformat()) \
            .lte("post_time", current_time_str) \
            .execute()

        return res.data or []
    except Exception as e:
        print(f"❌ Error fetching scheduled posts ready to post: {e}")
        return []

def create_post_reward_record(profile_id, post_id, platform, action_id=None):
    """Create initial post reward record when post is published"""
    try:
        # Set post_created_at to now and eligible_at to 7 days from now
        post_created_at = datetime.now(IST)
        eligible_at = post_created_at + timedelta(hours=168)

        reward_data = {
            "profile_id": profile_id,
            "post_id": post_id,
            "platform": platform,
            "reward_status": "pending",
            "post_created_at": post_created_at.isoformat(),
            "eligible_at": eligible_at.isoformat(),
            "reward_value": None
        }

        # TODO: Add action_id when column is available in schema
        # For now, action_id will be found from post_contents during reward calculation
        # if action_id:
        #     reward_data["action_id"] = action_id

        supabase.table("post_rewards").insert(reward_data).execute()
        print(f"reward data inserted successfully")
        print(f"reward data: {reward_data}")
    except Exception as e:
        print(f"Error creating post reward record for {post_id}: {e}")
        raise
def insert_action(post_id, platform, context, action):
    try:
        res = supabase.table("rl_actions").insert({
            "post_id": post_id,
            "platform": platform,
            "hook_type": action.get("HOOK_TYPE"),
            "information_depth": action.get("INFORMATION_DEPTH"),  # Changed from LENGTH to hook_length
            "tone": action.get("TONE"),
            "creativity": action.get("CREATIVITY"),
            "composition_style": action.get("COMPOSITION_STYLE"),
            "visual_style": action.get("VISUAL_STYLE"),
            "time_bucket": context.get("time_bucket"),
            "topic": None,  # Will be set from main.py
            "business_id": None  # Will be set from main.py
        }).execute()

        if res.data and len(res.data) > 0 and "id" in res.data[0]:
            return res.data[0]["id"]
        else:
            raise ValueError("Failed to insert action - no ID returned")
    except Exception as e:
        print(f"Error inserting action for post_id {post_id}: {e}")
        raise
def insert_post_snapshot(post_id, platform, metrics, profile_id=None, timeslot_hours=24):
    try:
        # Ensure metrics values are properly typed
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                processed_metrics[key] = value
            elif isinstance(value, str) and value.isdigit():
                processed_metrics[key] = int(value)
            else:
                processed_metrics[key] = value  # Keep as-is for strings/other types

        # Prepare data according to schema
        snapshot_data = {
            "profile_id": profile_id,  # Default business ID
            "post_id": post_id,
            "platform": platform,
            "timeslot_hours": timeslot_hours,
            "snapshot_at": datetime.now(IST).isoformat(),
            **processed_metrics
        }

        supabase.table("post_snapshots").insert(snapshot_data).execute()
    except Exception as e:
        print(f"Error inserting post snapshot for post_id {post_id}: {e}")
        raise

def update_baseline_ema(previous_baseline: float, current_reward: float, beta: float) -> float:
    """
    Reinforcement Learning utility for updating reward baseline using exponential moving average.

    PURPOSE: Normalizes rewards for stable RL learning by maintaining a running average of past rewards.

    FORMULA: b_{t+1} = (1 - β) · b_t + β · R_t

    Args:
        previous_baseline: b_t (current baseline value)
        current_reward: R_t (latest reward from post engagement)
        beta: β (baseline learning rate, typically 0.1, controls adaptation speed)

    Returns:
        updated_baseline: b_{t+1} (new baseline for next reward calculation)

    Example:
        update_baseline_ema(0.45, 0.72, 0.1) → 0.477
    """
    return (1 - beta) * previous_baseline + beta * current_reward

# Global baseline storage for mathematical baseline updates
_platform_baselines = {}

def get_platform_baseline(platform: str, default: float = 0.0) -> float:
    """Get current baseline for a platform"""
    return _platform_baselines.get(platform, default)

def set_platform_baseline(platform: str, baseline: float):
    """Set baseline for a platform"""
    _platform_baselines[platform] = baseline

def update_baseline_mathematical(platform: str, current_reward: float, beta: float = 0.1) -> float:
    """
    Pure mathematical baseline update using exponential moving average.

    Formula: b_{t+1} = (1 - β) · b_t + β · R_t

    This is the ONLY correct way for RL baseline calculation.
    """
    previous_baseline = get_platform_baseline(platform, 0.0)
    new_baseline = update_baseline_ema(previous_baseline, current_reward, beta)

    # Store the updated baseline
    set_platform_baseline(platform, new_baseline)

    print(f"📊 Mathematical baseline update for {platform}: {previous_baseline:.4f} → {new_baseline:.4f} (reward: {current_reward:.4f}, beta: {beta})")

    return new_baseline
def get_profile_embedding(profile_id):
    """Retrieve profile embedding from profiles table"""
    try:
        res = supabase.table("profiles") \
            .select("user_context_embedding") \
            .eq("id", profile_id) \
            .execute()

        if res.data and len(res.data) > 0:
            row = res.data[0]
            if "user_context_embedding" in row and row["user_context_embedding"] is not None:
                # user_context_embedding can be returned as a list/array or string from Supabase
                embedding_data = row["user_context_embedding"]

                if isinstance(embedding_data, list):
                    return np.array(embedding_data, dtype=np.float32)
                elif isinstance(embedding_data, str):
                    # Parse string representation of vector (e.g., "[1.0, 2.0, 3.0]" or "1.0,2.0,3.0")
                    try:
                        # Remove brackets if present and split by comma
                        cleaned_str = embedding_data.strip('[]')
                        values = [float(x.strip()) for x in cleaned_str.split(',')]
                        return np.array(values, dtype=np.float32)
                    except (ValueError, AttributeError) as parse_error:
                        print(f"Error parsing embedding string: {parse_error}")
                        return None
                else:
                    print(f"Unexpected embedding format: {type(embedding_data)}")
                    return None

        return None
    except Exception as e:
        print(f"Error retrieving profile embedding for {profile_id}: {e}")
        return None


def get_profile_embedding_with_fallback(profile_id):
    """Get profile embedding, return None if not found (no fake data)"""
    embedding = get_profile_embedding(profile_id)

    if embedding is not None:
        return embedding

    # No fallback to fake data - return None to indicate missing data
    print(f"⚠️ Profile embedding not found for {profile_id} - no embedding available")
    return None

def get_profile_business_data(profile_id):
    """
    Fetch normalized business context for prompts, embeddings, and RL.
    Always returns a complete, non-null dictionary.
    """
    try:
        res = supabase.table("profiles").select(
            """
            business_name,
            business_type,
            industry,
            business_description,
            brand_voice,
            brand_tone,
            target_audience,
            unique_value_proposition,
            customer_pain_points,
            primary_color,
            secondary_color,
            location_state,
            logo_url
            """
        ).eq("id", profile_id).execute()

        if res.data:
            p = res.data[0]

            return {
                # Core identity
                "business_name": p.get("business_name") or "Business",
                "business_types": p.get("business_type") or ["General"],
                "industries": p.get("industry") or ["General"],

                # Brand & messaging
                "business_description": p.get("business_description") or "A business focused on growth",
                "brand_voice": p.get("brand_voice") or "Professional and approachable",
                "brand_tone": p.get("brand_tone") or "Friendly and informative",

                # Audience & value
                "target_audience": p.get("target_audience") or ["23-45 years","Parents/Families","Business Owners/Entrepreneurs","Corporate Clients/B2B Buyers","Educators/Trainers","Freelancers/Creators","Tech Enthusiasts/Gamers","Impulse Buyers","Budget-Conscious Shoppers"],
                "unique_value_proposition": p.get("unique_value_proposition") or "We are a business that provides a service to our customers and helps them grow their business",
                "customer_pain_points": p.get("customer_pain_points") or "No pain points currently",

                # Visual & geo context
                "primary_color": p.get("primary_color") or "#000000",
                "secondary_color": p.get("secondary_color") or "#FFFFFF",
                "location_state": p.get("location_state") or "Gujarat",
                "logo_url": p.get("logo_url") or None
            }

    except Exception as e:
        print(f"Error fetching profile business data for {profile_id}: {e}")

    # 🔒 ABSOLUTE FALLBACK (never returns None anywhere)
    return {
        "business_name": "Business",
        "business_types": ["B2B","B2C"],
        "industries": ["General"],
        "business_description": "A business focused on growth",
        "brand_voice": "Professional and approachable",
        "brand_tone": "Friendly and informative",
        "target_audience": ["23-45 years","Parents/Families","Business Owners/Entrepreneurs","Corporate Clients/B2B Buyers","Educators/Trainers","Freelancers/Creators","Tech Enthusiasts/Gamers","Impulse Buyers","Budget-Conscious Shoppers"],
        "unique_value_proposition": "We are a business that provides a service to our customers and helps them grow their business",
        "customer_pain_points": "No pain points currently",
        "primary_color": "#000000",
        "secondary_color": "#FFFFFF",
        "location_state": "Gujarat"
    }


def get_profile_scheduling_prefs(profile_id):
    """Fetch user's preferred scheduling time from profiles table"""
    try:
        res = supabase.table("profiles").select(
            "time_bucket"
        ).eq("id", profile_id).execute()

        if res.data and len(res.data) > 0:
            profile = res.data[0]
            return {
                "time_bucket": profile.get("time_bucket")
            }

        # Fallback defaults if no data found
        return {
            "time_bucket": "evening"
        }
    except Exception as e:
        print(f"Error fetching profile scheduling preferences for {profile_id}: {e}")
        return {
            "time_bucket": "evening"
        }


def get_all_profile_ids():
    """
    Returns a list of active profile IDs from the profiles table.
    Only considers profiles with subscription_status = 'active'.
    Used for iterating over all active businesses in the system.
    """
    try:
        res = supabase.table("profiles").select("id").eq("subscription_status", "active").execute()

        if res.data:
            return [profile["id"] for profile in res.data]
        else:
            return []

    except Exception as e:
        print(f"Error fetching active profile IDs: {e}")
        return []


def should_create_post_today(profile_id) -> bool:
    """
    Checks if a post has already been created for today for the given profile.
    Returns True if no post exists for today, False otherwise.
    """
    try:
        current_date = datetime.now(IST).date().isoformat()
        
        # Query post_contents for any post by this business on this date
        res = supabase.table("post_contents") \
            .select("post_id") \
            .eq("business_id", profile_id) \
            .eq("post_date", current_date) \
            .execute()
            
        if res.data and len(res.data) > 0:
            return False  # Already posted today
            
        return True  # No post yet today
        
    except Exception as e:
        print(f"Error checking daily eligibility for {profile_id}: {e}")
        # Default to True to avoid missing posts on DB error, 
        # but in production you might prefer False to be safe.
        return True


def get_post_metrics(post_id, platform):
    """Fetch real metrics for a post from database"""
    try:
        res = supabase.table("post_snapshots") \
            .select("*") \
            .eq("post_id", post_id) \
            .eq("platform", platform) \
            .execute()

        if res.data and len(res.data) > 0:
            # Return the metrics dict, excluding metadata fields
            row = res.data[0]
            metrics = {k: v for k, v in row.items()
                      if k not in ['post_id', 'platform', 'created_at', 'id']}
            return metrics
        return None
    except Exception as e:
        print(f"Error fetching metrics for post {post_id}: {e}")
        return None



def get_post_reward(profile_id: str, post_id: str, platform: str):
    try:
        res = (
            supabase.table("post_rewards")
            .select("*")
            .eq("profile_id", profile_id)
            .eq("post_id", post_id)
            .eq("platform", platform)
            .single()
            .execute()
        )
        return res.data
    except Exception as e:
        print(f"Error fetching reward record: {e}")
        return None
def get_post_snapshots(profile_id: str, post_id: str, platform: str):
    res = (
        supabase.table("post_snapshots")
        .select("timeslot_hours, likes, comments, shares, saves, replies, retweets, reactions")
        .eq("profile_id", profile_id)
        .eq("post_id", post_id)
        .eq("platform", platform)
        .execute()
    )
    return res.data
def calculate_reward_from_snapshots(snapshots: list, platform: str, post_id: str = None) -> float:
    reward = 0.0
    print(f"Calculating reward for {platform} with {len(snapshots)} snapshots")

    # Check if post is deleted for penalty calculation
    deleted = False
    days_since_post = None

    if post_id:
        try:
            # Check post status in post_contents table
            post_result = supabase.table("post_contents").select("status, created_at").eq("post_id", post_id).eq("platform", platform).execute()
            if post_result.data and len(post_result.data) > 0:
                post_data = post_result.data[0]
                if post_data.get("status") == "deleted":
                    deleted = True
                    # Calculate days since post creation for penalty scaling
                    if post_data.get("created_at"):
                        created_at = datetime.fromisoformat(post_data["created_at"].replace('Z', '+00:00'))
                        current_time = datetime.now(IST)
                        if created_at.tzinfo is not None:
                            created_at = created_at.replace(tzinfo=None)
                        # Make current_time timezone-naive to match created_at
                        current_time = current_time.replace(tzinfo=None)
                        days_since_post = (current_time - created_at).days
                        print(f"   Post is deleted ({days_since_post} days ago), applying penalty")
        except Exception as e:
            print(f"   Could not check post deletion status: {e}")

    for snap in snapshots:
        t = snap["timeslot_hours"]
        weight = REWARD_WEIGHTS.get(t)

        if not weight:
            continue

        # Platform-specific engagement calculation using shared utility
        engagement = calculate_platform_engagement(platform, snap)

        # Apply time-based weighting
        weighted_engagement = weight * engagement
        reward += weighted_engagement
        print(f"   {t}h snapshot: {engagement:.2f} engagement x {weight} weight = {weighted_engagement:.4f}")

    # Apply normalization (log normalization with tanh bounding)
    followers = max(snapshots[0].get("follower_count", 1), 1) if snapshots else 1
    raw_score = math.log(1 + reward) / math.log(1 + followers)
    final_reward = math.tanh(raw_score)

    # -------------------------
    # 3️⃣ Delete penalty (human negative feedback)
    # -------------------------
    if deleted:
        # Deleted post = very strong negative signal
        if days_since_post is None or days_since_post == 0:
            penalty = 1.5  # Immediate deletion = maximum penalty
        else:
            # exponential decay penalty (stronger than before)
            penalty = 1.2 * math.exp(-days_since_post / 2.0)

        print(f"   Applying deletion penalty: -{penalty:.4f} (days_since_post: {days_since_post})")
        final_reward -= penalty

        # Ensure reward doesn't go below -1
        final_reward = max(final_reward, -1.0)

    print(f"   Total reward: {reward:.4f}, Followers: {followers}, Raw score: {raw_score:.4f}, Final reward: {final_reward:.4f}")

    return final_reward
def fetch_or_calculate_reward(profile_id: str, post_id: str, platform: str):
    print(f"Fetching/calculating reward for post {post_id} on {platform}")
    reward_row = get_post_reward(profile_id, post_id, platform)

    # Handle case where reward record doesn't exist yet
    if reward_row is None:
        print(f"   Reward record doesn't exist yet for {post_id}")
        return {
            "status": "pending",
            "reward": None
        }

    # 1️⃣ Already calculated → return immediately
    if reward_row.get("reward_status") == "calculated":
        existing_reward = reward_row.get("reward_value")
        print(f"   Reward already calculated: {existing_reward}")
        return {
            "status": "calculated",
            "reward": existing_reward
        }

    # 2️⃣ Check eligibility status (handle multiple valid states)
    status = reward_row.get("reward_status", "pending")
    if status == "pending":
        eligible_at = reward_row.get("eligible_at")
        if eligible_at:
            # Handle timezone-aware vs timezone-naive datetime comparison
            try:
                # Parse the eligible_at datetime and make it timezone-naive for comparison
                if eligible_at.endswith('Z'):
                    eligible_dt = datetime.fromisoformat(eligible_at[:-1])
                else:
                    eligible_dt = datetime.fromisoformat(eligible_at)
                # If it's timezone-aware, convert to naive UTC
                if eligible_dt.tzinfo is not None:
                    eligible_dt = eligible_dt.replace(tzinfo=None)

                current_dt = datetime.now(IST).replace(tzinfo=None)
                if current_dt < eligible_dt:
                    print(f"   Reward not yet eligible (eligible at: {eligible_at})")
                    return {
                        "status": "pending",
                        "reward": None
                    }
            except (ValueError, AttributeError) as e:
                # If parsing fails, assume it's not eligible
                print(f"   Could not parse eligible_at: {eligible_at} (error: {e})")
                return {
                    "status": "pending",
                    "reward": None
                }

    # 3️⃣ Eligible or eligible status → calculate ONCE
    print(f"   Calculating reward (status: {status})")
    snapshots = get_post_snapshots(profile_id, post_id, platform)

    if not snapshots:
        # No snapshots available yet
        print(f"   No snapshots available yet for {post_id}")
        return {
            "status": "pending",
            "reward": None
        }

    reward_value = calculate_reward_from_snapshots(snapshots, platform, post_id)

    try:
        print(f"   Updating reward record with calculated value: {reward_value}")
        supabase.table("post_rewards").update({
            "reward_status": "calculated",
            "reward_value": reward_value,
            "calculated_at": datetime.now(IST).isoformat()
        }).eq("id", reward_row["id"]).execute()

        # Also store final reward in rl_rewards table
        print(f"   📊 Storing reward in rl_rewards table")
        action_id = reward_row.get("action_id")

        # If action_id not in reward record, try to find it from post_contents
        if not action_id:
            try:
                post_content = supabase.table("post_contents").select("action_id").eq("post_id", reward_row["post_id"]).eq("platform", platform).execute()
                if post_content.data and len(post_content.data) > 0:
                    action_id = post_content.data[0].get("action_id")
                    print(f"   🔗 Found action_id from post_contents: {action_id}")
            except Exception as e:
                print(f"   ⚠️  Could not find action_id: {e}")

        if not action_id:
            print(f"   ⚠️  Warning: No action_id found, skipping rl_rewards insert")
        else:
            # Calculate and update platform baseline using pure mathematics
            current_baseline = update_baseline_mathematical(platform, reward_value, beta=0.1)

            supabase.table("rl_rewards").insert({
                "action_id": action_id,  # Link to rl_actions record
                "platform": platform,
                "reward_value": reward_value,
                "baseline": current_baseline,  # Now using actual calculated baseline
                "deleted": False,
                "days_to_delete": None,
                "reward_window": "24h"
            }).execute()
        print(f"   ✅ Reward calculation completed successfully")

    except Exception as e:
        print(f"Error updating reward: {e}")
        return {
            "status": "error",
            "reward": None
        }

    return {
        "status": "calculated",
        "reward": reward_value
    }


def get_connected_platforms(business_id):
    """
    Fetch all active connected platforms for a business
    
    Args:
        business_id (str): The business/user ID
        
    Returns:
        list: List of active platform names (e.g., ['instagram', 'linkedin', 'x'])
    """
    try:
        # Query platform_connections table for active connections
        res = supabase.table("platform_connections") \
            .select("platform") \
            .eq("user_id", business_id) \
            .eq("is_active", True) \
            .eq("connection_status", "active") \
            .execute()
        
        if res.data:
            platforms = [connection["platform"] for connection in res.data]
            print(f"📱 Found {len(platforms)} connected platforms: {platforms}")
            return platforms
        else:
            print(f"⚠️  No active platform connections found for business {business_id}")
            return []
            
    except Exception as e:
        print(f"❌ Error fetching connected platforms for {business_id}: {e}")
        return []


# ============================================
# STORAGE FUNCTIONS
# ============================================

def upload_image_to_storage(image_data: bytes, filename: str, bucket_name: str = "content_images", folder: str = "generated") -> str:
    """
    Saves raw bytes to Supabase and returns the public URL.
    Improved implementation based on best practices.

    Args:
        image_data: Raw image bytes
        filename: Base filename (will be made unique)
        bucket_name: Storage bucket name (default: "content_images")
        folder: Subfolder within bucket (default: "generated")

    Returns:
        Public URL of the uploaded image
    """
    import logging
    import uuid
    from datetime import datetime

    # Set up logging
    logger = logging.getLogger(__name__)

    try:
        # 1. Create a unique filename to prevent overwriting
        timestamp = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        unique_filename = f"{timestamp}_{unique_id}_{filename}"
        file_path = f"{folder}/{unique_filename}"

        # 2. Upload to Storage with proper options
        response = supabase.storage.from_(bucket_name).upload(
            path=file_path,
            file=image_data,
            file_options={
                "content-type": "image/png",
                "x-upsert": "false"  # Prevent overwriting existing files
            }
        )

        # 3. Handle possible errors
        if hasattr(response, 'error') and response.error:
            logger.error(f"Upload failed: {response.error}")
            raise RuntimeError(f"Upload failed: {response.error}")

        # 4. Construct and return Public URL
        public_url = supabase.storage.from_(bucket_name).get_public_url(file_path)
        logger.info(f"Successfully uploaded image: {public_url}")
        return public_url

    except Exception as e:
        logger.error(f"Exception during image upload: {str(e)}")
        raise RuntimeError(f"Failed to upload image: {str(e)}")


def upload_base64_image_to_storage(b64_string: str, filename: str, bucket_name: str = "content_images", folder: str = "generated") -> str:
    """
    Helper for AI models that return base64 encoded images.
    Decodes base64 and uploads to Supabase storage.

    Args:
        b64_string: Base64 encoded image string
        filename: Base filename (will be made unique)
        bucket_name: Storage bucket name
        folder: Subfolder within bucket

    Returns:
        Public URL of the uploaded image
    """
    import base64

    # Decode base64 to bytes
    try:
        image_bytes = base64.b64decode(b64_string)
        return upload_image_to_storage(image_bytes, filename, bucket_name, folder)
    except Exception as e:
        raise RuntimeError(f"Failed to decode base64 image: {str(e)}")


class SupabaseImageManager:
    """
    Image manager for Supabase Storage uploads.
    Provides a clean interface for image uploads with proper error handling.
    """

    def __init__(self, bucket_name: str = "content_images"):
        """
        Initialize the image manager.

        Args:
            bucket_name: Default bucket to use for uploads
        """
        import logging

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Load credentials
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.bucket_name = bucket_name

        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY missing!")

        self.supabase: Client = create_client(self.url, self.key)

    def save_image(
        self,
        image_data: bytes,
        filename: str,
        folder: str = "generated",
        content_type: str = "image/png"
    ) -> str:
        """
        Save raw image bytes to Supabase Storage and return public URL.

        Args:
            image_data: Raw image bytes
            filename: Base filename (will be made unique)
            folder: Subfolder within bucket
            content_type: MIME type of the image

        Returns:
            Public URL of the uploaded image
        """
        # Create unique filename to prevent overwriting
        timestamp = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        safe_filename = f"{timestamp}_{uid}_{filename}"
        file_path = f"{folder}/{safe_filename}"

        # Upload to Storage
        response = self.supabase.storage.from_(self.bucket_name).upload(
            path=file_path,
            file=image_data,
            file_options={
                "content-type": content_type,
                "upsert": False  # Prevent overwriting existing files
            }
        )

        # Handle errors - check different response formats
        if hasattr(response, 'error') and response.error:
            self.logger.error(f"Upload failed: {response.error}")
            raise RuntimeError(response.error)
        elif isinstance(response, dict) and response.get("error"):
            self.logger.error(f"Upload failed: {response['error']}")
            raise RuntimeError(response["error"])

        # Return public URL
        public_url = self.supabase.storage.from_(self.bucket_name).get_public_url(file_path)
        self.logger.info(f"Successfully uploaded: {public_url}")
        return public_url

    def save_base64_image(
        self,
        b64_string: str,
        filename: str,
        folder: str = "generated"
    ) -> str:
        """
        Decode base64 image and upload to Supabase.
        Handles both plain base64 and data URLs (with MIME prefix).

        Args:
            b64_string: Base64 encoded image string (may include data URL prefix)
            filename: Base filename (will be made unique)
            folder: Subfolder within bucket

        Returns:
            Public URL of the uploaded image
        """
        # Handle data URLs (remove prefix if present)
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        # Decode base64 to bytes
        image_bytes = base64.b64decode(b64_string)
        return self.save_image(image_bytes, filename, folder)
