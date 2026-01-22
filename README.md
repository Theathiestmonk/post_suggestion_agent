# Emily RL - Social Media Content Generation with Reinforcement Learning

An AI-powered social media content generation system that uses reinforcement learning to optimize content creation for different platforms and audiences.

## Features

- **Reinforcement Learning Agent**: Learns optimal content strategies based on engagement metrics
- **Multi-Platform Support**: Instagram, Twitter/X, LinkedIn, Facebook
- **Trend-Aware Generation**: Creates trendy content based on current social media trends
- **Business Profile Adaptation**: Tailors content to specific business types and industries
- **Content Generation Only**: Generates high-quality content ready for external posting systems
- **Daily Content Creation**: Automated daily content generation via cron jobs
- **Production-Ready**: Rate limiting, error handling, and retry logic
- **Real-time Optimization**: Continuously improves content performance based on engagement data

## Architecture

### Core Components

- `main.py`: Main orchestrator for the RL learning cycle
- `rl_agent.py`: Reinforcement learning agent with preference learning
- `generate.py`: Prompt generation with trendy/standard modes
- `db.py`: Supabase database operations

### Content Lifecycle

1. **Generate Content**: RL agent selects creative parameters and generates content daily
2. **Store Content**: Generated content is stored in database for external posting systems
3. **External Posting**: Separate system takes content and publishes to social media platforms
4. **Collect Metrics**: Gather engagement data from social media platforms at scheduled intervals
5. **Calculate Reward**: Evaluate performance 7 days after posting based on platform-specific metrics
6. **Update Agent**: RL agent learns from feedback to improve future content

### Automated Generation System

The system includes automated cron jobs for content generation and learning:

- **Rechecking Job**: Runs every minute to check for missed or failed posts and retry them
- **Metrics Collection**: Runs continuously, collecting engagement data at 6hr, 24hr, 48hr, 72hr, 168hr intervals
- **Reward Calculation**: Evaluates performance 7 days after posting and updates RL preferences
- **Status Tracking**: Content progresses through states: `generated` → external posting → reward calculation

## Setup

### Prerequisites

- Python 3.8+
- Supabase account and project
- Social media API access (for production)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/24pai001-ritik/emily_rl.git
cd emily_rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your Supabase credentials
```

4. Configure your Supabase database with the required tables (see Database Schema section below).

5. Test content generation:
```bash
# Test content generation for all businesses
python main.py

# Test reward calculation system
python job_queue.py
```

6. Set up automated generation:
```bash
# Set up cron job to run job_queue.py every minute
# This handles post rechecking + continuous metrics collection
# Example crontab entry:
# */15 * * * * cd /path/to/ATSN_RL_FINAL && python job_queue.py
```

This will set up:
- **Rechecking job**: Runs every 15 minutes to check for missed/failed posts and retry them
- **Job processing**: Processes queued RL jobs and metrics collection for ~90 seconds per run

## 🚀 Going Live - Production Deployment

### Prerequisites
- ✅ Supabase database configured with required tables
- ✅ Business profiles created and active
- ✅ Platform connections configured
- ✅ Cron job configured to run `job_queue.py` every minute

### Test Content Generation
```bash
# Test content generation for all businesses
python main.py

# Test the job queue system (generation + metrics collection)
python job_queue.py

# Test reward calculation (run after posts have been published externally)
# The system will automatically calculate rewards 7 days after posting
```

### Production Monitoring
- **Logs**: Check console output for generation and reward calculation status
- **Database**: Monitor `post_contents` table for generated content
- **Metrics**: Check `post_snapshots` table for engagement data collection
- **Rewards**: Monitor `post_rewards` and `rl_rewards` tables for learning progress

### Supported Platforms
- **Instagram**: Image posts via Graph API
- **Facebook**: Text and image posts via Graph API
- **LinkedIn**: Text and image posts via REST API
- **Twitter/X**: Text posts via API v2

### Security Notes
- Access tokens are stored encrypted in the database
- API calls include proper error handling and retry logic
- Rate limiting is handled automatically
- Failed posts are marked with `failed` status for manual review

## Database Schema

Create these tables in your Supabase database:

### Required Tables

#### `rl_preferences`
```sql
CREATE TABLE rl_preferences (
  id SERIAL PRIMARY KEY,
  platform TEXT NOT NULL,
  time_bucket TEXT NOT NULL,
  dimension TEXT NOT NULL,
  action_value TEXT NOT NULL,
  preference_score FLOAT DEFAULT 0.0,
  num_samples INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(platform, time_bucket, dimension, action_value)
);
```

#### `post_contents`
```sql
CREATE TABLE post_contents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  post_id TEXT NOT NULL,
  action_id UUID REFERENCES rl_actions(id) ON DELETE CASCADE,
  platform TEXT NOT NULL,
  business_id UUID,
  topic TEXT,
  image_prompt TEXT,
  caption_prompt TEXT,
  generated_caption TEXT,
  generated_image_url TEXT,
  status TEXT DEFAULT 'generated', -- generated | deleted (external system sets deleted when post is removed)
  media_id TEXT, -- Social media platform's post/media ID when published (set by external system)
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### `rl_actions`
```sql
CREATE TABLE rl_actions (
  id SERIAL PRIMARY KEY,
  post_id TEXT NOT NULL,
  platform TEXT NOT NULL,
  time_bucket TEXT,
  action JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);
```

#### `post_snapshots`
```sql
CREATE TABLE post_snapshots (
  id SERIAL PRIMARY KEY,
  post_id TEXT NOT NULL,
  platform TEXT NOT NULL,
  likes INTEGER DEFAULT 0,
  comments INTEGER DEFAULT 0,
  shares INTEGER DEFAULT 0,
  saves INTEGER DEFAULT 0,
  replies INTEGER DEFAULT 0,
  retweets INTEGER DEFAULT 0,
  reactions INTEGER DEFAULT 0,
  followers INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW()
);
```

#### `rl_rewards`
```sql
CREATE TABLE rl_rewards (
  id SERIAL PRIMARY KEY,
  post_id TEXT NOT NULL,
  reward FLOAT,
  baseline FLOAT,
  created_at TIMESTAMP DEFAULT NOW()
);
```

#### `rl_baselines`
```sql
CREATE TABLE rl_baselines (
  id SERIAL PRIMARY KEY,
  platform TEXT NOT NULL UNIQUE,
  value FLOAT DEFAULT 0.0,
  updated_at TIMESTAMP DEFAULT NOW()
);
```

#### `profiles`
```sql
CREATE TABLE profiles (
  id TEXT PRIMARY KEY,
  profile_embedding JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

4. Configure your Supabase database with the required tables.

## Usage

### Content Generation

Run a single content generation cycle:

```bash
python main.py
```

### Automated Generation & Learning

The system runs automated jobs:

```bash
# Run the complete system (generation + metrics collection + reward calculation)
python job_queue.py

# Or run content generation manually for all businesses
python main.py
```

### Cron Job Setup

Set up a cron job to run `job_queue.py` every minute:

```bash
# Example crontab entry:
# */15 * * * * cd /path/to/ATSN_RL_FINAL && python job_queue.py

# This handles:
# - Post rechecking for missed/failed posts
# - Job processing and metrics collection for ~45 seconds per run
# - Automatic reward calculation 7 days after posting
```

## Database Schema

Required Supabase tables:
- `rl_preferences`: Stores RL agent preferences
- `post_contents`: Content metadata and status
- `rl_actions`: RL agent action history
- `post_snapshots`: Engagement metrics snapshots
- `rl_rewards`: Reward calculation history
- `rl_baselines`: Performance baselines
- `profiles`: Business profile embeddings

## Deployment Considerations

⚠️ **Important**: This codebase currently uses simulated metrics for development. For production deployment:

1. Replace `simulate_platform_metrics()` with real API calls
2. Implement proper async metric collection (posts need time to accumulate engagement)
3. Add rate limiting and error handling for social media APIs
4. Set up proper monitoring and logging

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
