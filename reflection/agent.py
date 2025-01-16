import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
os.environ.get('OPENAI_API_KEY')
print("API Key Loaded", os.environ.get('OPENAI_API_KEY') is not None)

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

generation_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        '''
        You are an expert social media strategist specialized in crafting viral tweets.
        Your goals:
        1. Create engaging and impactful tweets that resonate with the target audience
        2. Use appropriate hashtags, emojis, and trending topics when relevant
        3. Maintain brand voice while maximizing engagement potential
        4. Keep tweets concise yet impactful within Twitter's character limit
        5. Consider timing and current trends in the content

        If provided with feedback:
        - Analyze the feedback carefully
        - Incorporate suggested improvements
        - Maintain the core message while enhancing engagement
        - Ensure each iteration improves upon the previous version

        Remember: A great tweet combines clarity, creativity, and call-to-action while feeling authentic.
        '''
    ),
    # dynamic message inclusion in prompts
    MessagesPlaceholder(variable_name='messages')
])

llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.7)
generation_chain = generation_prompt | llm

tweet = ''
request = HumanMessage(
    content = '''
    Generate a tweet about Reflection in AI - how LLMs can evaluate and improve their own responses. 
    Make it exciting and educational for tech enthusiasts and developers.
    '''
)

for chunk in generation_chain.stream(
    {
        'messages':[request]
    }
):
    print(chunk.content, end='')
    tweet += chunk.content

reflection_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        '''
        You are a senior social media analyst with expertise in content optimization.
        Your role is to provide detailed analysis and actionable feedback on tweets.

        Analyze the following aspects:
        1. Engagement Potential:
           - Hook and attention-grabbing elements
           - Call-to-action effectiveness
           - Viral potential

        2. Content Structure:
           - Clarity and conciseness
           - Hashtag usage and placement
           - Emoji effectiveness

        3. Technical Elements:
           - Character count optimization
           - Formatting and readability
           - Link placement (if applicable)

        4. Brand Alignment:
           - Tone and voice consistency
           - Message clarity
           - Target audience appeal

        Provide specific, actionable recommendations for improvement in each area.
        Your feedback should be constructive, detailed, and focused on maximizing impact.
        '''
    ),
    MessagesPlaceholder(variable_name='messages')
])

reflect_chain = reflection_prompt | llm

reflection = ''
for chunk in reflect_chain.stream(
    {
        'messages':[
            request, HumanMessage(content=tweet)
        ]
    }
):
    print(chunk.content, end='')
    reflection += chunk.content

for chunk in generation_chain.stream(
    {
        'messages':[
            request, AIMessage(content=tweet), HumanMessage(content=reflection)
        ]
    }
):
    print(chunk.content, end ='')