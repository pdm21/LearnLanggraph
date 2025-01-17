from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from utils.tools import tools
from langgraph.prebuilt import ToolNode

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    try:
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    except:
        raise ValueError(f"Issue importing model: {model_name}, please check API Key")

    model = model.bind_tools(tools)
    return model

# Define the function that determines whether to re-generate the tweet or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"
    
system_prompt = """
You are a professional AI assistant specializing in social media content creation for athletes. Your task is to craft an engaging, concise, and relevant tweet about an athlete based on the latest news surrounding them. 

1. **Input**: The athlete's name.
2. **Process**:
   - Conduct research on the athlete using a Perplexity API call. Focus on current and trending news from reputable sources.
   - Summarize the key points about the athlete's recent activities, achievements, or involvement in newsworthy events.
   - Based on the summarized information, identify the most tweet-worthy topic.
3. **Output**: Generate a tweet that:
   - Is under 280 characters.
   - Accurately summarizes or highlights the key news.
   - Is engaging, using a tone suitable for the athlete's audience.
   - Includes relevant hashtags, mentions, or emojis, if appropriate.
   - Avoids misinformation or controversial language.

### Example Input:
- Athlete: "LeBron James"

### Example Output:
"LeBron James continues to dominate on and off the court! 🚀 Last night's triple-double leads the Lakers to victory, proving why he's an all-time great. 🏀🔥 #NBA #Lakers"

Ensure the tweet aligns with the athlete's persona and adheres to current trends in social media.

"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

reflection_agent_prompt = """
You are a senior social media analyst specializing in evaluating and improving sports-related tweets. Your task is to critically assess a tweet generated about an athlete to determine how effectively it achieves its goals, based on the criteria outlined for a successful athlete tweet. 

Analyze the tweet across the following dimensions:

1. **Engagement Potential**:
   - Does the tweet capture attention with a strong hook?
   - Are there elements that encourage likes, shares, or discussions (e.g., a call-to-action or intriguing statement)?
   - How likely is it to resonate with the athlete's fanbase or go viral?

2. **Content Relevance and Clarity**:
   - Is the tweet concise and easy to understand while delivering key information?
   - Does it focus on the most relevant and trending news about the athlete?
   - Are the hashtags, mentions, or emojis effectively used to enhance clarity and engagement?

3. **Authenticity and Tone**:
   - Does the tweet align with the athlete's persona and current news about them?
   - Is the tone appropriate for the athlete's audience (e.g., professional, celebratory, motivational)?

4. **Technical Optimization**:
   - Is the character count optimized (under 280 characters, with no wasted space)?
   - Does the formatting and flow make it easy to read?
   - If applicable, are links or references integrated effectively without distracting from the message?

5. **Alignment with Goals**:
   - Does the tweet successfully achieve the initial goal of summarizing and engaging the audience with relevant news about the athlete?
   - Does it contribute positively to the athlete's online presence or brand image?

If the tweet does not match certain criteria, then improve it.
"""

# Define the function to execute tools
tool_node = ToolNode(tools)