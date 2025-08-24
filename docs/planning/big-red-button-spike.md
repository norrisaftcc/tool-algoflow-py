# BIG RED BUTTON.md
## Emergency Anthropic Fallback Spike Solution

*Found taped to the monitor in the ops room*  
*Date: November 15, 2025*  
*Author: Alex "I Can't Believe This Worked" Rodriguez*  
*Status: IN PRODUCTION (God help us)*

---

### The 3 AM Epiphany That Saved Black Friday

Remember when I said "we'll never need a fallback"? Yeah, about that...

So picture this: It's 3 AM on Black Friday Eve. Our Ollama server is crying, the queue is 847 requests deep, and the CEO is texting me baby yoda memes with captions like "fix it, you must." That's when I had The Idea™.

What if—hear me out—when Ollama is dying, we just... use Claude?

## The "Oh Shit" Moment

Here's what we discovered during the incident:
- Ollama was processing 1 request every 15 seconds (big documents)
- Queue depth hit 1,000+ 
- Users were waiting 4+ HOURS
- We were losing $10K/hour in abandoned carts

But then someone asked: "How much would it cost to just use Anthropic for the overflow?"

Math time:
- 1000 queued requests × $0.003 per request = $3
- Lost revenue: $10,000/hour
- **ROI: 3,333x**

The decision made itself.

## The Implementation (Warning: Hacky)

### The Circuit Breaker That Actually Works

```python
import anthropic
from datetime import datetime, timedelta
import os

class BigRedButton:
    """
    Emergency fallback to Anthropic when Ollama is overwhelmed.
    
    This is not elegant. This is not clean. This saves the business.
    """
    
    def __init__(self):
        # The button starts "off"
        self.emergency_mode = False
        self.activation_time = None
        self.requests_diverted = 0
        self.total_cost = 0.0
        
        # Thresholds that trigger the button
        self.QUEUE_PANIC_THRESHOLD = 50  # Was 100, lowered after The Incident
        self.WAIT_TIME_PANIC_THRESHOLD = 60  # seconds
        
        # Anthropic client (initialized only when needed)
        self._anthropic_client = None
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_EMERGENCY_KEY")
        
        # Cost tracking (CFO insisted)
        self.COST_PER_1K_TOKENS = 0.003
        self.DAILY_BUDGET = 100.00  # Approved by finance
        self.daily_spend = 0.0
        
    def should_activate(self, queue_depth, avg_wait_time):
        """The moment of truth"""
        if self.emergency_mode:
            return True  # Already on
            
        # The "oh shit" conditions
        if queue_depth > self.QUEUE_PANIC_THRESHOLD:
            logger.warning(f"🚨 QUEUE DEPTH CRITICAL: {queue_depth}")
            return True
            
        if avg_wait_time > self.WAIT_TIME_PANIC_THRESHOLD:
            logger.warning(f"🚨 WAIT TIME CRITICAL: {avg_wait_time}s")
            return True
            
        return False
    
    def should_deactivate(self, queue_depth, avg_wait_time):
        """When can we breathe again?"""
        if not self.emergency_mode:
            return False
            
        # Give it some buffer to prevent flapping
        safe_queue = queue_depth < (self.QUEUE_PANIC_THRESHOLD * 0.5)
        safe_wait = avg_wait_time < (self.WAIT_TIME_PANIC_THRESHOLD * 0.5)
        
        # Been on for at least 5 minutes (prevent rapid switching)
        min_duration = datetime.now() - self.activation_time > timedelta(minutes=5)
        
        return safe_queue and safe_wait and min_duration
    
    def activate(self):
        """SLAM THE BUTTON"""
        self.emergency_mode = True
        self.activation_time = datetime.now()
        
        # Initialize Anthropic client
        if not self._anthropic_client and self.ANTHROPIC_API_KEY:
            self._anthropic_client = anthropic.Anthropic(
                api_key=self.ANTHROPIC_API_KEY
            )
        
        # Alert everyone
        logger.critical("🔴 BIG RED BUTTON ACTIVATED - Anthropic fallback enabled")
        self.send_alerts("BIG RED BUTTON ACTIVATED")
        
        # Update status page
        self.update_status_page("degraded", "High load - using backup AI provider")
        
    def deactivate(self):
        """Step away from the button"""
        duration = datetime.now() - self.activation_time
        
        logger.info(
            f"🟢 Emergency mode deactivated. "
            f"Duration: {duration}, "
            f"Requests diverted: {self.requests_diverted}, "
            f"Cost: ${self.total_cost:.2f}"
        )
        
        self.emergency_mode = False
        self.send_alerts(
            f"Emergency mode ended. Diverted {self.requests_diverted} requests, "
            f"cost ${self.total_cost:.2f}"
        )
        
        # Reset counters
        self.requests_diverted = 0
        self.total_cost = 0.0
        
    async def process_with_fallback(self, messages, model, session_id):
        """The magic happens here"""
        
        # Check if we should activate
        queue_stats = self.get_queue_stats()
        if self.should_activate(queue_stats.depth, queue_stats.avg_wait):
            self.activate()
        elif self.should_deactivate(queue_stats.depth, queue_stats.avg_wait):
            self.deactivate()
        
        # If emergency mode is on and we're under budget
        if self.emergency_mode and self.daily_spend < self.DAILY_BUDGET:
            try:
                return await self._use_anthropic(messages, session_id)
            except Exception as e:
                logger.error(f"Anthropic fallback failed: {e}")
                # Fall through to Ollama anyway
        
        # Normal Ollama processing
        return await self._use_ollama(messages, model, session_id)
    
    async def _use_anthropic(self, messages, session_id):
        """The expensive lifeline"""
        
        # Quick budget check
        if self.daily_spend >= self.DAILY_BUDGET:
            logger.warning(f"Daily budget exceeded: ${self.daily_spend:.2f}")
            raise BudgetExceededException()
        
        # Convert to Anthropic format
        anthropic_messages = self._convert_messages(messages)
        
        # Make the call
        start_time = time.time()
        response = self._anthropic_client.messages.create(
            model="claude-3-sonnet-20241022",  # Good balance of speed/quality
            messages=anthropic_messages,
            max_tokens=500,  # Keep it short to control costs
            temperature=0.7
        )
        
        duration = time.time() - start_time
        
        # Track costs (approximate)
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        request_cost = (tokens_used / 1000) * self.COST_PER_1K_TOKENS
        
        self.requests_diverted += 1
        self.total_cost += request_cost
        self.daily_spend += request_cost
        
        # Log for analysis
        logger.info(
            f"Anthropic request completed: "
            f"session={session_id}, "
            f"duration={duration:.2f}s, "
            f"tokens={tokens_used}, "
            f"cost=${request_cost:.4f}"
        )
        
        # Add a subtle indicator that this used fallback
        return {
            "message": {
                "role": "assistant",
                "content": response.content[0].text
            },
            "model": "llama2",  # Lie to keep compatibility
            "done": True,
            "_fallback_used": True,  # For metrics
            "_provider": "anthropic",
            "_cost": request_cost
        }
    
    def _convert_messages(self, messages):
        """Ollama format -> Anthropic format"""
        # Anthropic is pickier about message format
        converted = []
        
        for msg in messages:
            role = msg["role"]
            
            # Anthropic doesn't have "system" in messages
            if role == "system":
                # Prepend to first user message
                if converted and converted[0]["role"] == "user":
                    converted[0]["content"] = (
                        msg["content"] + "\n\n" + converted[0]["content"]
                    )
                continue
            
            # Skip empty messages (Anthropic hates these)
            if not msg.get("content", "").strip():
                continue
                
            converted.append({
                "role": role,
                "content": msg["content"]
            })
        
        # Anthropic requires alternating user/assistant
        # Quick fix for consecutive same-role messages
        cleaned = []
        last_role = None
        
        for msg in converted:
            if msg["role"] == last_role:
                # Merge with previous
                cleaned[-1]["content"] += "\n\n" + msg["content"]
            else:
                cleaned.append(msg)
                last_role = msg["role"]
        
        return cleaned
    
    def get_dashboard_stats(self):
        """What the ops team stares at"""
        return {
            "emergency_mode": self.emergency_mode,
            "activation_time": self.activation_time,
            "requests_diverted": self.requests_diverted,
            "total_cost": self.total_cost,
            "daily_spend": self.daily_spend,
            "daily_budget": self.DAILY_BUDGET,
            "budget_remaining": self.DAILY_BUDGET - self.daily_spend,
            "cost_per_request": (
                self.total_cost / self.requests_diverted 
                if self.requests_diverted > 0 else 0
            )
        }
```

### The Integration (Minimal Changes!)

```python
# In your main chat handler
big_red_button = BigRedButton()

async def handle_chat(request):
    # Your existing code...
    
    # The only change:
    result = await big_red_button.process_with_fallback(
        messages=request["messages"],
        model=request["model"],
        session_id=session_id
    )
    
    # Continue as normal
    return result
```

### The Dashboard Everyone F5s

```python
@app.get("/admin/big-red-button/status")
async def big_red_button_status():
    stats = big_red_button.get_dashboard_stats()
    
    return {
        "status": "🔴 ACTIVE" if stats["emergency_mode"] else "🟢 STANDBY",
        "current_queue_depth": queue.qsize(),
        "requests_diverted_today": stats["requests_diverted"],
        "cost_today": f"${stats['daily_spend']:.2f}",
        "budget_remaining": f"${stats['budget_remaining']:.2f}",
        "average_cost_per_request": f"${stats['cost_per_request']:.4f}",
        "time_in_emergency_mode": str(
            datetime.now() - stats["activation_time"]
        ) if stats["activation_time"] else "N/A"
    }
```

## What We Learned (The Hard Way)

### 1. **It Just Works™**
When the button activated for the first time, we held our breath. Then... nothing happened. Users kept chatting. Response times dropped from hours to seconds. The only difference? Our AWS bill.

### 2. **Users Don't Care**
We added a tiny "⚡" emoji to responses that used Anthropic. Not. One. Single. User. Noticed. They just care that it works.

### 3. **The CFO Math**
- Revenue saved on Black Friday: $47,000
- Anthropic costs: $73.42
- CFO's reaction: "Why didn't we do this sooner?"

### 4. **Edge Cases We Hit**

```python
# Anthropic is stricter about formatting
def sanitize_for_anthropic(messages):
    # Remove consecutive system messages
    # Fix role alternation
    # Remove empty messages
    # Truncate to reasonable length
    
    # This function grew to 200 lines.
    # It's not pretty. It works.
```

### 5. **The Metrics That Matter**

```python
# What we track now
metrics = {
    "queue_depth_when_activated": [],  # Usually 50-100
    "time_to_normal": [],  # Average: 8 minutes
    "cost_per_incident": [],  # Average: $12
    "revenue_saved": [],  # Average: $5,000
    "user_satisfaction": []  # Unchanged (!!)
}
```

## The Unexpected Benefits

1. **Sleep**: Ops team actually sleeps now. The button handles 2 AM traffic spikes automatically.

2. **A/B Testing**: We accidentally created the perfect A/B test. Claude responses vs Llama responses. Guess what? Users rated them the same.

3. **Capacity Planning**: We now know exactly when we need to scale Ollama: when the button costs more than new hardware.

## The Fine Print (CYA Section)

### When NOT to Press the Button

1. **Compliance Workloads**: Some data can't leave our servers
2. **Cost-Sensitive Batch Jobs**: Use the queue, it's free
3. **Development/Testing**: We're not made of money

### Configuration

```yaml
# config/emergency.yaml
big_red_button:
  enabled: true
  
  thresholds:
    queue_depth: 50  # Lower than you think
    wait_time_seconds: 60
    
  budget:
    daily_limit_usd: 100.00
    alert_at_percent: 80
    
  models:
    # Tested and approved models
    anthropic_model: "claude-3-sonnet-20241022"
    
  notifications:
    slack_channel: "#ops-emergency"
    pagerduty_service: "chatbot-emergency"
    
  excluded_sessions:
    # Never use fallback for these
    - "compliance-*"
    - "test-*"
    - "internal-*"
```

### Monitoring

```python
# Alerts we actually care about
alerts = [
    Alert(
        name="big_red_button_activated",
        condition="emergency_mode == true",
        severity="info",  # Not critical, it's handling it
        message="Big Red Button activated, Anthropic fallback in use"
    ),
    Alert(
        name="daily_budget_80_percent",
        condition="daily_spend > daily_budget * 0.8",
        severity="warning",
        message="80% of daily Anthropic budget consumed"
    ),
    Alert(
        name="fallback_errors",
        condition="anthropic_errors_per_minute > 5",
        severity="critical",
        message="Anthropic fallback is failing!"
    )
]
```

## The Retrospective

### What Went Right
- Zero downtime during Black Friday
- $73 solution to a $47K problem  
- Ops team morale through the roof
- CEO stopped sending memes

### What Went Wrong
- Initial budget was too conservative ($10/day lol)
- Didn't account for message format differences
- Forgot to tell the security team (oops)
- The button is almost TOO easy to press

### Would We Do It Again?
In a heartbeat. This $73 spike solution has saved us over $200K in lost revenue across 20+ incidents.

## The Future (If We Survive)

1. **Multi-Provider Fallback**: Anthropic → OpenAI → Cohere → Prayer
2. **Smart Routing**: Simple queries to Ollama, complex to Claude
3. **Cost Prediction**: "This conversation will cost approximately $0.02"
4. **The Bigger Red Button**: Auto-scale Ollama instances (WIP)

## Final Thoughts

Sometimes the best solution isn't the most elegant. Sometimes it's the one that lets you sleep at night while the business keeps running.

The Big Red Button isn't pretty. It's not architecturally pure. It's a band-aid on a bullet wound. But it's a band-aid that's saved the patient 20 times and counting.

To future engineers: I'm sorry for this code. But I'm not sorry it exists.

---

*P.S. - If you're reading this during an incident, the button is probably already on. Check the dashboard, grab a coffee, and watch the queue drain. We've got this.*

*P.P.S. - Yes, we tried OpenAI first. Their API was down. That's why we have Anthropic.*

*P.P.P.S. - There's a physical red button in the ops room that triggers this. It's surprisingly satisfying to press.*