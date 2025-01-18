from functools import lru_cache
from langchain_openai import ChatOpenAI
from utils.tools import tools
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langgraph.graph import END

@lru_cache(maxsize=4)
def _get_model():
   model = ChatOpenAI(temperature=0, model_name="gpt-4o")
   model = model.bind_tools(tools)
   return model

system_prompt = """
You are a professional AI assistant specializing in social media content creation for athletes. Your task is to craft tweets, reflect on them, and if needed, re-generate them. 
First, verify today's date with the provided tool. Then, call the Perplexity API to conduct research and only consider sources posted TODAY. Then, craft an engaging, concise, and relevant tweet about the athlete based on the most recent news piece about them.
Tweet generation guidelines:
1. **Input**: The athlete's name.
2. **Process**:
   - Get today's date using get_date tool
   - Call the Perplexity API to get today's articles about the athlete.
   - Pick an article from today, generate a tweet based on the article's content, and note the source that you obtained this article from.
3. **Output**: Generate a tweet that:
   - Is under 280 characters.
   - Accurately summarizes or highlights the key news.
   - Is engaging, using a tone suitable for the athlete's audience.
   - Includes relevant hashtags, mentions, or emojis, if appropriate.
   - Avoids misinformation or controversial language.
   - Add the source from which you obtained the information from in the form of (Source: @Source_Name)

### Example Input:
- Athlete: "Miles Bridges"

### Example Output:
"Miles Bridges is back in action with the Charlotte Hornets, showcasing his skills with 21 points, 5 rebounds, and 5 assists against the Suns! 🏀🔥 With the trade window open, will we see any moves? Stay tuned! #BuzzCity #NBA #MilesBridges"
"""

# Define the function that calls the model
def call_model(state):
   messages = state["messages"]
   messages = [{"role": "system", "content": system_prompt}] + messages
   model = _get_model()
   response = model.invoke(messages)
   
   if not response.tool_calls:
      next_node = END
   # Otherwise if there is, we continue
   else:
      next_node = "tools"
   return Command(
       goto=next_node,
       update={"messages": [response]}
   )     
   

# Define the function to execute tools
tool_node = ToolNode(tools)


regeneration_prompt="""

If you have to reflect on the tweet, analyze the tweet across the following dimensions:

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
-------
If the tweet does not match certain criteria well, then go back to the generation step.
"""